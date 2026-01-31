import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from datetime import datetime, timezone, timedelta 
import random
import string
import re
import json
import hashlib
from functools import wraps
import easyocr
import io 
import cv2 
import numpy as np 
import google.generativeai as genai 
from flask_mail import Mail, Message 
from face_verification import verify_face_match
from dotenv import load_dotenv
import hashlib
from functools import wraps
import bcrypt
from cryptography.fernet import Fernet
import base64
import os

load_dotenv()

print("DEBUG USER:", os.environ.get('MAIL_USERNAME'))
print("DEBUG PASS:", os.environ.get('MAIL_PASSWORD'))

# try:
#     from document_extracter import extract_text_from_file 
# except ImportError:
#     print("Warning: 'document_extracter' not found. OCR simulation will be used.")
#     def extract_text_from_file(*args, **kwargs):
#         raise NotImplementedError("document_extracter module is missing.")

# EasyOCR Reader 
try:
    ocr_reader = easyocr.Reader(['en']) 
    print("EasyOCR reader loaded successfully (English Only).")
except Exception as e:
    print(f"Warning: Could not load EasyOCR. OCR route will fail. Error: {e}")
    ocr_reader = None

# Configuration & File Setup 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(BASE_DIR, 'voters.json') 
VOTES_FILE = os.path.join(BASE_DIR, 'votes.json') 
ELECTIONS_FILE = os.path.join(BASE_DIR, 'elections.json') 

SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_secret_key_for_truecast_sessions') 
ENCRYPTION_KEY = os.environ.get('DATA_ENCRYPTION_KEY', 'default_insecure_key_32_bytes_long_')

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Email Configuration
# For Gmail, you must use an "App Password", not your regular password.
# Go to Google Account > Security > 2-Step Verification > App Passwords.
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

mail = Mail(app)

def hash_pin(pin):
    """Hashes a plaintext PIN using bcrypt."""
    return bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_pin(pin, hashed_pin):
    """Checks a plaintext PIN against the stored hash."""
    if not hashed_pin:
        return False
    return bcrypt.checkpw(pin.encode('utf-8'), hashed_pin.encode('utf-8'))

def hash_record(data_dict):
    """
    Creates a deterministic SHA-256 hash of a dictionary.
    Sorting keys ensures the hash is consistent regardless of insertion order.
    """
    serialized_data = json.dumps(
        data_dict,
        sort_keys=True, 
        separators=(',', ':') 
    ).encode('utf-8')    
    return hashlib.sha256(serialized_data).hexdigest()

def get_last_hash():
    """
    FIXED: Reads the last vote record from the ledger and returns its 'current_hash'.
    Removes the broken reference to '.cache' to fix the AttributeError.
    """
    votes_data = load_votes()
    if not votes_data:
        return "0000000000000000000000000000000000000000000000000000000000000000"

    try:
        last_vote = votes_data[next(reversed(votes_data))]
    except StopIteration:
        return "0000000000000000000000000000000000000000000000000000000000000000"
    return last_vote.get('current_hash', "0000000000000000000000000000000000000000000000000000000000000000")

try:
    key_bytes = base64.urlsafe_b64encode(ENCRYPTION_KEY.encode().ljust(32)[:32])
    cipher = Fernet(key_bytes)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Fernet cipher. Check DATA_ENCRYPTION_KEY: {e}")
    cipher = None 

def encrypt_data(data):
    """Encrypts a string for storage."""
    if not cipher or not data: return data
    try:
        return cipher.encrypt(data.encode()).decode()
    except Exception:
        return "ENCRYPTION_FAILED"

def decrypt_data(data):
    """Decrypts a stored string."""
    if not cipher or not data: return data
    try:
        return cipher.decrypt(data.encode()).decode()
    except Exception:
        return data 

@app.after_request
def add_security_headers(response):
    """
    Prevents the browser from caching secure pages after logout.
    This forces the browser to re-request the page from the server, 
    where the login_required check will run.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache" 
    response.headers["Expires"] = "0" 
    return response

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

TRUECAST_SYSTEM_PROMPT = """
ROLE: You are the official TRUECAST Support AI. Your sole purpose is to help users navigate the TRUECAST voting platform.

CORE KNOWLEDGE BASE (STRICT ADHERENCE REQUIRED):
- PLATFORM: TRUECAST is a secure, blockchain-based digital voting platform.
- REGISTRATION STEPS: 1. Click "Register to Vote". 2. Enter personal details. 3. Upload ID (Aadhaar, Voter ID/EPIC, PAN, or Driving License). 4. Perform Face Verification.
- LOGIN METHODS: Users can login using 1. Voter ID + Backup PIN, 2. Face Recognition, or 3. Email OTP.
- SECURITY: Votes are stored on an immutable blockchain ledger. This provides a digital "Audit Trail" similar to VVPAT.
- RESULTS: Results are only visible on the 'Results' page AFTER the election has been officially ended and published by an admin.

STRICT OPERATIONAL RULES:
1. ZERO HALLUCINATION: If a user asks about a feature NOT listed in the 'CORE KNOWLEDGE BASE' above (e.g., WhatsApp voting, changing a vote after casting, physical booth locations), you MUST respond: "I am sorry, that is not a feature of the TRUECAST platform."
2. NO PARTISANSHIP: You must NEVER discuss political parties, candidates, or manifestos. If asked who to vote for, say: "As an AI, I am strictly neutral. Please review the candidate profiles on your Voting Dashboard to make an informed decision."
3. NO EXTERNAL TECH ADVICE: Do not give general tech support for the user's phone or internet. Stick to the TRUECAST app interface.
4. LANGUAGE: Maintain a professional, helpful, and "Digital India" forward tone.
5. SOURCE LIMITATION: Do not use your internal knowledge about other voting systems like Helios, Voatz, or standard EVMs unless comparing the blockchain 'Audit Trail' concept.

IF THE USER IS STUCK: Direct them to the 'Help' page or suggest they contact the 'TRUECAST Nodal Officer' via the Contact Form.
"""

try:
    chat_model = genai.GenerativeModel(
        model_name='gemini-2.5-flash', 
        system_instruction=TRUECAST_SYSTEM_PROMPT
    )
    print("Gemini AI initialized with: gemini-2.5-flash (Stable)")
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize model gemini-2.5-flash. Details: {e}")
    if not GEMINI_API_KEY:
        print("ACTION REQUIRED: GEMINI_API_KEY is missing. Check your .env file.")
    chat_model = None

@app.route('/api/send-otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    voter_identifier = data.get('voterId') 
    if not voter_identifier:
        return jsonify({'success': False, 'error': 'Voter ID is required.'}), 400
    target_voter = get_voter_by_identifier(voter_identifier)  
    if not target_voter:
        return jsonify({'success': False, 'error': 'Voter not found.'}), 404    
    encrypted_email = target_voter.get('email')
    email = decrypt_data(encrypted_email)
    if not email or email == "ENCRYPTION_FAILED" or not re.match(r'[^@]+@[^@]+\.[^@]+', email):
        return jsonify({'success': False, 'error': 'No valid email address linked to this voter.'}), 400
    otp = str(random.randint(100000, 999999))
    session['otp'] = otp
    session['otp_voter_id'] = target_voter['voter_id']
    session['otp_timestamp'] = datetime.now(timezone.utc).timestamp()
    try:
        msg = Message('TrueCast Login Verification', recipients=[email])
        msg.body = f"Your One-Time Password (OTP) for TrueCast voting is: {otp}\n\nThis code expires in 5 minutes.\nDo not share this code."
        mail.send(msg)
        masked_email = re.sub(r'(.).*@', r'\1***@', email)
        return jsonify({'success': True, 'message': f'OTP sent to {masked_email}'})
    except Exception as e:
        print(f"Email error: {e}")
        return jsonify({'success': False, 'error': 'Failed to send email. Check server logs.'}), 500

@app.route('/api/verify-otp-login', methods=['POST'])
def verify_otp_login():
    data = request.get_json()
    input_otp = data.get('otp')
    stored_otp = session.get('otp')
    stored_voter_id = session.get('otp_voter_id')
    timestamp = session.get('otp_timestamp')
    if not stored_otp or not input_otp:
        return jsonify({'success': False, 'error': 'Invalid request.'}), 400
    if datetime.now(timezone.utc).timestamp() - timestamp > 300:
        session.pop('otp', None)
        return jsonify({'success': False, 'error': 'OTP has expired. Please request a new one.'}), 400
    if input_otp == stored_otp:
        voters = load_voters()
        authenticated_voter = voters.get(stored_voter_id)
        if authenticated_voter:
            session['logged_in'] = True
            session['voter_id'] = authenticated_voter['voter_id']
            session['email'] = authenticated_voter.get('email')
            session['voter_region'] = authenticated_voter.get('voterRegion')
            first = authenticated_voter.get('firstName', '')
            last = authenticated_voter.get('lastName', '')
            session['full_name'] = f"{first} {last}".strip() or 'Voter'
            session.pop('otp', None)
            session.pop('otp_voter_id', None)
            session.pop('otp_timestamp', None)
            return jsonify({
                'success': True, 
                'message': 'Authentication successful!',
                'redirect': url_for('voting_dashboard')
            })
    return jsonify({'success': False, 'error': 'Invalid OTP.'}), 401

IST = timezone(timedelta(hours=5, minutes=30))

def load_voters():
    """Reads the voter data from the JSON file."""
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w') as f:
             json.dump({}, f)
        return {}
    try:
        with open(JSON_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(JSON_FILE, 'w') as f:
             json.dump({}, f)
        return {} 

def save_voters(data):
    """Writes the voter data back to the JSON file."""
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
def load_votes():
    """Reads the dictionary of cast votes from the JSON file."""
    if not os.path.exists(VOTES_FILE):
        with open(VOTES_FILE, 'w') as f:
             json.dump({}, f) 
        return {}
    try:
        with open(VOTES_FILE, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, FileNotFoundError):
        with open(VOTES_FILE, 'w') as f:
             json.dump({}, f)
        return {}

def save_votes(data):
    """Writes the vote dictionary back to the JSON file."""
    with open(VOTES_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_elections():
    """Reads the list of elections from the JSON file."""
    if not os.path.exists(ELECTIONS_FILE):
        with open(ELECTIONS_FILE, 'w') as f:
             json.dump([], f) 
        return []
    try:
        with open(ELECTIONS_FILE, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, FileNotFoundError):
        with open(ELECTIONS_FILE, 'w') as f:
             json.dump([], f)
        return []

def save_elections(data):
    """Writes the election list back to the JSON file."""
    with open(ELECTIONS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_active_election():
    """
    Returns the currently active election, or None, and automatically updates
    the status of expired elections.
    """
    elections = load_elections()
    now = datetime.now(IST)
    elections_changed = False
    active_election = None
    for election in elections:
        try:
            start_time = datetime.fromisoformat(election.get('startDate'))
            end_time = datetime.fromisoformat(election.get('endDate'))
            if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=IST)
            if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=IST)
        except (ValueError, TypeError):
            continue 
        status = election.get('status', 'Active')
        if status == 'Active' and end_time <= now:
            election['status'] = 'Ended'
            elections_changed = True
        elif election['status'] == 'Active' and now < end_time:
            active_election = election
    if elections_changed:
        save_elections(elections)
    return active_election
    
    # # Fallback to the most recent 'Active' election if the time window is missed
    # for election in reversed(elections):
    #     if election.get('status') == 'Active':
    #         return election
    # return None

def get_election_by_id(election_id):
    """Finds an election by its ID."""
    elections = load_elections()
    for election in elections:
        if election.get('id') == election_id:
            return election
    return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('voter_login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('You must be an admin to access this page.', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function 

def generate_hash_id(length=64):
    return '0x' + ''.join(random.choices(string.hexdigits.lower(), k=length))

def preprocess_image(file_bytes):
    """
    Cleans the image for better OCR results:
    1. Convert to Grayscale
    2. Apply Thresholding (Binarization) to make text pop
    3. Denoise
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return file_bytes 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, encoded_img = cv2.imencode('.jpg', thresh)
    return encoded_img.tobytes()

def clean_text_keep_english(text):
    """
    Removes non-ASCII characters to confuse the regex parser less.
    """
    return re.sub(r'[^\x00-\x7F]+', '', text)

def parse_ocr_text(text):
    """
    Generic Pattern-Based Parser for Indian IDs.
    """
    text = clean_text_keep_english(text)
    parsed_data = {}
    aadhaar_match = re.search(r'\b(\d{4}\s?\d{4}\s?\d{4})\b', text)
    pan_match = re.search(r'\b([A-Z]{5}\d{4}[A-Z])\b', text)
    passport_match = re.search(r'\b([A-Z]\d{7})\b', text)
    dl_match = re.search(r'\b([A-Z]{2}[-\s]?\d{13,})\b', text)
    if aadhaar_match:
        parsed_data['ID Number'] = aadhaar_match.group(1).replace(" ", "")
        parsed_data['docType'] = 'Aadhaar Card'
    elif pan_match:
        parsed_data['ID Number'] = pan_match.group(1)
        parsed_data['docType'] = 'PAN Card'
    elif passport_match:
        parsed_data['ID Number'] = passport_match.group(1)
        parsed_data['docType'] = 'Passport'
    elif dl_match:
        parsed_data['ID Number'] = dl_match.group(1)
        parsed_data['docType'] = 'Driving License'
    else:
        parsed_data['ID Number'] = 'Not Found'
        parsed_data['docType'] = 'Unknown'
    dob_match = re.search(r'\b(\d{2}[/-]\d{2}[/-](?:19|20)\d{2})\b', text)
    if dob_match:
        parsed_data['Date of Birth'] = dob_match.group(1)
    else:
        yob_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if yob_match:
             parsed_data['Date of Birth'] = "01/01/" + yob_match.group(1)
        else:
             parsed_data['Date of Birth'] = 'Not Found'
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name_found = False
    if parsed_data['docType'] == 'Aadhaar Card':
        for i, line in enumerate(lines):
            if "DOB" in line or "Year of Birth" in line or "Birth" in line:
                if i > 0:
                    potential_name = lines[i-1]
                    if "GOVERNMENT" not in potential_name.upper() and not any(char.isdigit() for char in potential_name):
                        parsed_data['Full Name'] = potential_name
                        name_found = True
                        break
            elif parsed_data['Date of Birth'] != 'Not Found' and parsed_data['Date of Birth'] in line:
                 if i > 0:
                    potential_name = lines[i-1]
                    if "GOVERNMENT" not in potential_name.upper() and not any(char.isdigit() for char in potential_name):
                        parsed_data['Full Name'] = potential_name
                        name_found = True
                        break
    if not name_found:
        stop_words = ["GOVERNMENT", "INDIA", "MALE", "FEMALE", "DOB", "DATE", "BIRTH", "ADDRESS", 
                      "YEAR", "FATHER", "HUSBAND", "NAME", "CARD", "INCOME", "TAX", "DEPARTMENT",
                      "UNIQUE", "IDENTIFICATION", "AUTHORITY", "PERMANENT", "ACCOUNT", "NUMBER"]              
        potential_names = []
        for line in lines:
            clean_line = line.strip()
            if not clean_line or any(char.isdigit() for char in clean_line) or len(clean_line) < 4:
                continue  
            words = clean_line.split()
            if any(word.upper() in stop_words for word in words):
                continue  
            if 2 <= len(words) <= 4 and all(word.isalpha() for word in words):
                 potential_names.append(clean_line)
        if potential_names:
            parsed_data['Full Name'] = potential_names[0]
        else:
            parsed_data['Full Name'] = 'Not Found'
    pin_match = re.search(r'\b(\d{6})\b', text)
    if pin_match:
        pin_code = pin_match.group(1)
        pin_index = text.find(pin_code)
        start_index = max(0, pin_index - 120)
        raw_addr_text = text[start_index:pin_index + 6]
        clean_addr = re.sub(r'(Address|To|S/O|W/O|C/O)\s*[:.-]?\s*', '', raw_addr_text, flags=re.IGNORECASE)
        clean_addr = re.sub(r'\n', ', ', clean_addr)
        parsed_data['Address'] = clean_addr.strip()
    else:
        parsed_data['Address'] = 'Not Found'
    return parsed_data

@app.route('/')
def home():
    return render_template('truecast_landing.html',
                           logged_in=session.get('logged_in', False),
                           full_name=session.get('full_name', ''))

@app.route('/api/ocr_process', methods=['POST'])
def ocr_process():
    if 'idDocument' not in request.files:
        return jsonify({"success": False, "error": "No document uploaded."}), 400
    file = request.files.get('idDocument') 
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No document selected."}), 400
    if not ocr_reader:
         return jsonify({"success": False, "error": "OCR service is not available."}), 500
    try:
        file_bytes = file.read()
        processed_bytes = preprocess_image(file_bytes)
        ocr_result = ocr_reader.readtext(processed_bytes, detail=0, paragraph=True)
        raw_text = " \n ".join(ocr_result)
        print("--- OCR RAW TEXT (Processed) ---")
        print(raw_text)
        print("--------------------------------")
        parsed_data = parse_ocr_text(raw_text)
        print("--- PARSED DATA (English Pattern) ---")
        print(parsed_data)
        print("-------------------------------------")
        session['ocr_data'] = parsed_data
        return jsonify({
            "success": True, 
            "parsed_data": parsed_data,
            "raw_text": raw_text 
        })
    except Exception as e:
        print(f"OCR Processing Error: {e}") 
        return jsonify({"success": False, "error": f"An error occurred during OCR processing: {e}"}), 500

@app.route('/voter-register', methods=['GET', 'POST'])
def voter_register():
    active_election = get_active_election()
    available_regions = {
        'North District', 'South District', 'East District', 
        'West District', 'Central District'
    }
    if active_election:
        for race in active_election.get('races', []):
            for candidate in race.get('candidates', []):
                r_val = candidate.get('region')
                if r_val and r_val != 'All Regions':
                    available_regions.add(r_val)
    sorted_regions = sorted(list(available_regions))    
    parsed_data = {}
    ocr_results = []
    if request.method == 'POST':
        data = request.form.to_dict()
        new_email = data.get('email', '').strip() 
        voter_photo_b64 = data.get('voterPhotoBase64')
        voters = load_voters()
        for voter_id, voter_data in voters.items():
            stored_email = voter_data.get('email', '').strip()
            if stored_email and stored_email == new_email:
                flash('Registration failed: An account with this email address already exists. Please log in.', 'error')
                return render_template(
                    'truecast_voter_register.html',
                    parsed_data=data,
                    available_regions=sorted_regions
                ) 
        raw_pin = data.get('backupPin', '000000')
        hashed_pin = hash_pin(raw_pin) 
        raw_answer = data.get('securityAnswer', '')
        if raw_answer:
            data['securityAnswer'] = hash_pin(raw_answer)
        data['idNumber'] = encrypt_data(data.get('idNumber', ''))
        data['email'] = encrypt_data(new_email)
        data['phone'] = encrypt_data(data.get('phone', ''))
        data['address'] = encrypt_data(data.get('address', '')) 
        ocr_data = session.get('ocr_data', {})
        if ocr_data:
            is_valid, error_message = validate_registration(data, ocr_data)
            if not is_valid:
                flash(f'Registration Warning: {error_message}', 'warning')
        voter_id = f"VS{datetime.now().year}{random.randint(100000, 999999)}"
        data['voter_id'] = voter_id
        data['registration_date'] = datetime.now(timezone.utc).isoformat() 
        data['status'] = 'Active' 
        data['backupPin'] = hashed_pin
        data['registration_photo'] = voter_photo_b64
        voters[voter_id] = data
        save_voters(voters)
        session.pop('ocr_data', None) 
        flash(f'Registration successful! Your new Voter ID is {voter_id}. Please log in.', 'success')
        return redirect(url_for('voter_login', success='true'))
    session.pop('ocr_data', None)
    return render_template(
        'truecast_voter_register.html',
        parsed_data=parsed_data,
        ocr_results=ocr_results,
        available_regions=sorted_regions
    )

def validate_registration(form_data, ocr_data):
    """
    Helper function to validate form data against OCR session data.
    """
    ocr_name = ocr_data.get('Full Name', 'Not Found').lower()
    if ocr_name == 'not found':
        print("Warning: Name not found in OCR, proceeding with manual entry trust.")
        pass 
    form_first = form_data.get('firstName', '').lower()
    form_last = form_data.get('lastName', '').lower()
    if ocr_name != 'not found':
        if form_first not in ocr_name and form_last not in ocr_name:
             return False, f"Name on form ('{form_first} {form_last}') does not match name on ID ('{ocr_name}')."
    ocr_address = ocr_data.get('Address', 'Not Found').lower()
    if ocr_address == 'not found':
        if ocr_data.get('docType') == 'Aadhaar Card':
             return False, "Could not read address from ID document. Please try again or use a clearer image."
        else:
            pass 
    form_region = form_data.get('voterRegion')
    region_map = {
        # North India
        'delhi': 'North District',
        'new delhi': 'North District',
        'punjab': 'North District',
        'haryana': 'North District',
        'chandigarh': 'North District',
        'himachal pradesh': 'North District',
        'jammu': 'North District',
        'kashmir': 'North District',
        'uttarakhand': 'North District',
        'uttar pradesh': 'North District', 
        # South India
        'karnataka': 'South District',
        'bengaluru': 'South District',
        'bangalore': 'South District',
        'tamil nadu': 'South District',
        'chennai': 'South District',
        'kerala': 'South District',
        'kochi': 'South District',
        'telangana': 'South District',
        'hyderabad': 'South District',
        'andhra pradesh': 'South District',
        'vizag': 'South District',
        # East India
        'west bengal': 'East District',
        'kolkata': 'East District',
        'odisha': 'East District',
        'bhubaneswar': 'East District',
        'bihar': 'East District',
        'jharkhand': 'East District',
        'assam': 'East District',
        'guwahati': 'East District',
        # West India
        'maharashtra': 'West District',
        'mumbai': 'West District',
        'pune': 'West District',
        'gujarat': 'West District',
        'ahmedabad': 'West District',
        'rajasthan': 'West District',
        'jaipur': 'West District',
        'pali': 'West District',
        'goa': 'West District',
        # Central India
        'madhya pradesh': 'Central District',
        'bhopal': 'Central District',
        'indore': 'Central District',
        'chhattisgarh': 'Central District',
        'raipur': 'Central District'
    }
    expected_region = 'All Regions' 
    for keyword, region in region_map.items():
        if keyword in ocr_address:
            expected_region = region
            break
    if form_region != expected_region and form_region != 'All Regions':
         return False, f"The address on your ID (in '{ocr_address}') suggests you are in '{expected_region}', but you selected '{form_region}'. Please select the correct region."
    return True, "Success"

@app.route('/api/verify-face-login', methods=['POST'])
def verify_face_login():
    data = request.get_json()
    voter_identifier = data.get('voterId')
    login_photo_b64 = data.get('loginPhotoBase64')
    if not voter_identifier or not login_photo_b64:
        print("Face Verification Error: Missing ID or photo data in request.")
        return jsonify({'success': False, 'error': 'Missing ID or photo data in request.'}), 400
    target_voter = get_voter_by_identifier(voter_identifier)
    if not target_voter:
        print(f"Face Verification Error: Voter ID {voter_identifier} not found in database.")
        return jsonify({'success': False, 'error': 'Voter not found.'}), 404
    registration_photo_b64 = target_voter.get('registration_photo')
    if not registration_photo_b64:
        print(f"Face Verification Error: Reference photo missing for voter {voter_identifier}.")
        return jsonify({'success': False, 'error': 'No reference photo found for this voter. Please use PIN or OTP.'}), 400
    try:
        match, message, score = verify_face_match(registration_photo_b64, login_photo_b64)
        print(f"--- BIOMETRIC VERIFICATION RESULT ---")
        print(f"Voter: {voter_identifier}, Match: {match}, Score: {score}")
        print(f"Message: {message}")
        print(f"-------------------------------------")
        if match:
            session['logged_in'] = True
            session['voter_id'] = target_voter['voter_id']
            session['email'] = decrypt_data(target_voter.get('email'))
            session['voter_region'] = target_voter.get('voterRegion')
            first = target_voter.get('firstName', '')
            last = target_voter.get('lastName', '')
            session['full_name'] = f"{first} {last}".strip() or 'Voter'
            return jsonify({
                'success': True, 
                'message': f'Face verified! Score: {score}. Login successful.',
                'redirect': url_for('voting_dashboard')
            })
        else:
            return jsonify({'success': False, 'error': f'Verification failed (Score: {score}). {message}'}), 401
    except Exception as e:
        print(f"Face Verification FATAL Error: {e}")
        return jsonify({'success': False, 'error': 'Internal verification error. Check server logs for details.'}), 500
@app.route('/voter-login', methods=['GET', 'POST'])
def voter_login():
    if 'logged_in' in session:
        return redirect(url_for('voting_dashboard'))
    if request.method == 'POST':
        data = request.get_json()
        voter_id_or_email = data.get('voterId')
        input_pin = data.get('backupPin')
        voters = load_voters()
        authenticated_voter = None
        for voter_id, voter_data in voters.items():
            stored_voter_id = voter_data.get('voter_id')
            if stored_voter_id == voter_id_or_email:
                authenticated_voter = voter_data
                break
            encrypted_email = voter_data.get('email')
            decrypted_email = decrypt_data(encrypted_email)
            if decrypted_email == voter_id_or_email:
                authenticated_voter = voter_data
                break
        if authenticated_voter:
            stored_hash = authenticated_voter.get('backupPin')
            if input_pin:
                if not stored_hash or not check_pin(input_pin, stored_hash):
                    return jsonify({"success": False, "error": "Invalid PIN provided. Access denied."}), 401                
            elif not input_pin and 'backupPin' in data:
                 pass 
            session['logged_in'] = True 
            session['voter_id'] = authenticated_voter['voter_id']
            session['email'] = decrypt_data(authenticated_voter.get('email'))
            first_name = authenticated_voter.get('firstName', '')
            last_name = authenticated_voter.get('lastName', '')
            if first_name or last_name:
                clean_full_name = f"{first_name} {last_name}".strip()
                session['full_name'] = clean_full_name
            else:
                session['full_name'] = authenticated_voter.get('Full Name', 'Voter')
            session['voter_region'] = authenticated_voter.get('voterRegion')
            next_url = request.args.get('next') or url_for('voting_dashboard')
            return jsonify({"success": True, "message": "Login successful!", "redirect": next_url})
        else:
            return jsonify({"success": False, "error": "Voter ID or Email not found."}), 404    
    success_msg = request.args.get('success')
    return render_template('truecast_voter_login.html', success=success_msg)

def get_voter_by_identifier(identifier):
    """
    Looks up a voter by unencrypted voter_id or by decrypting all stored emails.
    Returns the voter data dict or None.
    """
    voters = load_voters()
    for voter_data in voters.values():
        if voter_data.get('voter_id') == identifier:
            return voter_data
        encrypted_email = voter_data.get('email')
        decrypted_email = decrypt_data(encrypted_email)
        if decrypted_email == identifier:
            return voter_data
    return None

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('voter_id', None)
    session.pop('email', None)
    session.pop('full_name', None) 
    session.pop('voter_region', None) 
    return redirect(url_for('home'))

@app.route('/voting-dashboard', methods=['GET', 'POST'])
@login_required
def voting_dashboard():
    voter_id = session.get('voter_id')
    voter_region = session.get('voter_region')    
    active_election = get_active_election()
    all_elections = load_elections()
    if not active_election:
        return render_template('truecast_voting_dashboard.html', 
                               voter_id=voter_id,
                               full_name=session.get('full_name', 'Voter'),
                               voter_region=voter_region,
                               has_voted=False,
                               election_active=False,
                               election={},              
                               filtered_ballot=[],      
                               all_race_ids=[],        
                               all_elections=all_elections, 
                               previous_votes={})
    filtered_ballot = []
    final_required_races_list = []
    for race in active_election.get('races', []):
        race_copy = race.copy()
        race_copy['candidates'] = []
        for candidate in race.get('candidates', []):
            candidate_region = candidate.get('region')
            if candidate_region == voter_region or candidate_region == 'All Regions':
                race_copy['candidates'].append(candidate)
        if race_copy['candidates']:
            filtered_ballot.append(race_copy)
            final_required_races_list.append(race_copy['name']) 
    voters_data = load_voters()
    voter_info = voters_data.get(voter_id)
    if not voter_info:
        flash('Voter information could not be loaded.', 'error')
        return redirect(url_for('login'))
    full_name = f"{voter_info.get('firstName', '')} {voter_info.get('lastName', '')}"
    votes_data = load_votes()
    active_election_id = active_election['id'] if active_election else 'N/A'
    if request.method == 'POST':
        data = request.get_json()
        selections = data.get('selections')
        has_voted = voter_info.get('vote_status', {}).get(active_election_id, False)
        if has_voted:
            return jsonify({'success': False, 'error': 'Your vote has already been cast for this election.', 'transactionHash': 'ALREADY_CAST'})
        required_race_slugs = [name.replace(' ', '-').lower() for name in final_required_races_list]        
        for race_slug in required_race_slugs:
            if not selections.get(race_slug):
                original_race_name = next((r['name'] for r in filtered_ballot if r['name'].replace(' ', '-').lower() == race_slug), race_slug)
                return jsonify({'success': False, 'error': f'Please make a selection for the {original_race_name} race.'})
        anonymous_token = hashlib.sha256(os.urandom(32)).hexdigest()
        previous_hash = get_last_hash()
        vote_record = {
            'electionId': active_election_id,
            'timestamp': datetime.now(IST).isoformat(),
            'previous_hash': previous_hash,       
            **selections
        }
        transaction_hash = hash_record(vote_record)
        vote_record['current_hash'] = transaction_hash
        votes_data[anonymous_token] = vote_record 
        save_votes(votes_data)
        if 'vote_status' not in voter_info:
            voter_info['vote_status'] = {}
        if 'receipts' not in voter_info:
            voter_info['receipts'] = {}   
        voter_info['vote_status'][active_election_id] = True
        encrypted_hash = encrypt_data(transaction_hash)
        voter_info['receipts'][active_election_id] = encrypted_hash 
        save_voters(voters_data) 
        return jsonify({'success': True, 'transactionHash': transaction_hash})
    has_voted = voter_info.get('vote_status', {}).get(active_election_id, False)
    previous_votes = {}
    if has_voted:
        encrypted_hash = voter_info.get('receipts', {}).get(active_election_id)
        tx_hash = decrypt_data(encrypted_hash)
        if tx_hash and tx_hash != "ENCRYPTION_FAILED":
             previous_votes = {'transactionHash': tx_hash} 
    return render_template(
        'truecast_voting_dashboard.html',
        voter_id=voter_id,
        full_name=full_name,
        voter_region=voter_region,
        election_active=True,
        election=active_election,
        all_elections=all_elections, 
        filtered_ballot=filtered_ballot,
        all_race_ids=final_required_races_list, 
        has_voted=has_voted,
        previous_votes=previous_votes
    )

def verify_vote_chain_integrity():
    """
    AUDIT TOOL: Iterates through all vote records to verify the cryptographic chain of custody.
    Returns: (bool, str message)
    """
    votes_data = load_votes()
    if not votes_data:
        return True, "No votes to verify (chain is empty)."
    records = list(votes_data.values())
    for i in range(1, len(records)):
        current_record = records[i]
        previous_record = records[i - 1]
        expected_previous_hash = previous_record.get('current_hash')
        actual_previous_hash = current_record.get('previous_hash')
        if actual_previous_hash != expected_previous_hash:
            return False, f"Chain break detected at record #{i}! Expected hash: {expected_previous_hash}, Found: {actual_previous_hash}"
        record_for_hashing = current_record.copy()
        record_for_hashing.pop('current_hash', None) 
        recalculated_hash = hash_record(record_for_hashing)
        if recalculated_hash != current_record.get('current_hash'):
            return False, f"Integrity break detected at record #{i}! Recalculated hash does not match stored hash."
    return True, "Chain integrity verified successfully."

@app.route('/admin/end-election/<string:election_id>', methods=['POST'])
@admin_required
def end_election(election_id):
    elections = load_elections()
    found = False
    for i, election in enumerate(elections):
        if election.get('id') == election_id:
            elections[i]['status'] = 'Ended'
            elections[i]['endDate'] = datetime.now(IST).isoformat() 
            found = True
            break
    if found:
        save_elections(elections)
        flash(f'Election "{election_id}" has been officially ENDED. Results are now ready for review.', 'success')
    else:
        flash(f'Election ID {election_id} not found.', 'error')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/publish-results/<string:election_id>', methods=['POST'])
@admin_required
def publish_results(election_id):
    elections = load_elections()
    found = False
    for i, election in enumerate(elections):
        if election.get('id') == election_id:
            elections[i]['published_results'] = True
            found = True
            break
    if found:
        save_elections(elections)
        flash(f'Results for "{election_id}" have been PUBLISHED to the voters.', 'success')
    else:
        flash(f'Election ID {election_id} not found.', 'error')
    return redirect(url_for('admin_dashboard'))

@app.route('/vote-verification', methods=['GET', 'POST'])
def vote_verification():
    return render_template('truecast_vote_verification.html')

@app.route('/api/verify_vote', methods=['POST'])
def verify_vote():
    data = request.get_json()
    query = data.get('query')
    all_votes = load_votes()
    vote_record = None
    voter_id_found = "Voter ID is kept anonymous for security."
    for anonymous_key, vote_details in all_votes.items():
        if isinstance(vote_details, dict) and vote_details.get('current_hash') == query:
            vote_record = vote_details
            break
    if not vote_record and query.startswith('VS'): 
        voters_data = load_voters()
        voter_info = voters_data.get(query)
        if voter_info:
            latest_receipt_encrypted = next(iter(reversed(list(voter_info.get('receipts', {}).values()))), None)
            if latest_receipt_encrypted:
                latest_receipt_hash = decrypt_data(latest_receipt_encrypted) 
                for anonymous_key, vote_details in all_votes.items():
                    if vote_details.get('current_hash') == latest_receipt_hash:
                        vote_record = vote_details
                        voter_id_found = query
                        break
    if vote_record:
        election_id = vote_record.get('electionId')
        election_name = "N/A"
        if election_id:
            election = get_election_by_id(election_id)
            if election:
                election_name = election.get('title', f"Election {election_id}")
        stored_timestamp = vote_record.get('timestamp')
        display_timestamp = stored_timestamp if stored_timestamp else datetime.now(IST).isoformat()
        final_transaction_hash = vote_record.get('current_hash', 'N/A')
        demo_vote_data = {
            "voterId": voter_id_found,
            "election": election_name,
            "timestamp": display_timestamp,
            "status": "Confirmed",
            "transactionHash": final_transaction_hash,
            "blockNumber": random.randint(10000, 20000), 
            "confirmations": random.randint(100, 500),
            "gasUsed": random.randint(20000, 30000),
            "networkFee": round(random.uniform(0.001, 0.005), 3),
            "blockHash": generate_hash_id()
        }
        return jsonify({"success": True, "data": demo_vote_data})
    return jsonify({"success": False, "error": "Vote not found."}), 404

@app.route('/geo-verification')
def geo_verification():
    return render_template('truecast_geo_verification.html')

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'password': 
            session['admin_logged_in'] = True
            flash('Admin login successful!', 'success')
            return redirect('/admin-dashboard') 
        else:
            flash('Invalid admin credentials.', 'error')
    return render_template('truecast_admin_login.html')

@app.route('/admin-dashboard')
@admin_required
def admin_dashboard():
    voters_data = load_voters()
    votes_data = load_votes()
    elections = load_elections() 
    active_election = get_active_election()
    total_voters = len(voters_data)
    total_votes_cast = len(votes_data) 
    active_election_votes = 0
    active_election_races = 0 
    active_election_candidates = 0
    if active_election:
        active_election_id = active_election['id']
        active_election_races = len(active_election.get('races', [])) 
        active_election_candidates = sum(
            len(race.get('candidates', [])) 
            for race in active_election.get('races', [])
        )
        for vote in votes_data.values():
                if isinstance(vote, dict) and vote.get('electionId') == active_election_id:
                    active_election_votes += 1
    turnout = (total_votes_cast / total_voters * 100) if total_voters > 0 else 0
    election_vote_counts = {}
    for vote in votes_data.values():
        election_id = vote.get('electionId')
        if election_id:
            election_vote_counts[election_id] = election_vote_counts.get(election_id, 0) + 1            
    elections_for_template = []
    for election in elections:
        election_id = election.get('id')
        election['total_votes'] = election_vote_counts.get(election_id, 0) # Attach votes
        elections_for_template.append(election)
    try:
        recent_registrations = sorted(
            voters_data.values(), 
            key=lambda v: v.get('registration_date', '1970-01-01'), 
            reverse=True
        )[:5]
    except Exception:
        recent_registrations = list(voters_data.values())[:5]
    results = {}
    voted_count = total_votes_cast
    not_voted_count = total_voters - total_votes_cast
    turnout_data = {
        'labels': ['Votes Cast', 'Eligible Voters (Not Voted)'],
        'data': [voted_count, not_voted_count]
    }
    bar_chart_data = {'labels': ['No Races'], 'datasets': []} 
    if votes_data:
        for vote in votes_data.values():
            if isinstance(vote, dict):
                for race_slug, candidate_slug in vote.items():
                    if race_slug not in ['transactionHash', 'electionId', 'timestamp', 'previous_hash', 'current_hash']:
                        race_display = race_slug.replace('-', ' ').title()
                        race_tally = results.setdefault(race_display, {})
                        race_tally[candidate_slug] = race_tally.get(candidate_slug, 0) + 1 
        if results:
            top_race_slug = next(iter(results.keys()), None)
            if top_race_slug:
                race_data = results[top_race_slug]
                bar_chart_data['labels'] = list(race_data.keys())
                bar_chart_data['datasets'].append({
                    'label': top_race_slug,
                    'data': list(race_data.values()),
                    'backgroundColor': ['#3498db', '#27ae60', '#f1c40f', '#e74c3c'] 
                })
    return render_template(
        'truecast_admin_dashboard.html',
        total_voters=total_voters,
        total_votes_cast=total_votes_cast,
        turnout=turnout,
        active_election_races=active_election_races, 
        active_election_candidates=active_election_candidates,
        recent_registrations=recent_registrations,
        all_voters=voters_data.values(),
        elections=elections_for_template, 
        all_elections_list=elections,
        active_election=active_election,
        active_election_votes=active_election_votes,
        results=results,
        turnout_data=turnout_data,
        bar_chart_data=bar_chart_data
    )

@app.route('/admin/create-election', methods=['GET', 'POST'])
@admin_required
def create_election():
    if request.method == 'POST':
        form_data = request.form.to_dict(flat=False)
        elections = load_elections()
        start_raw = request.form.get('startDate')
        end_raw = request.form.get('endDate')        
        start_dt = datetime.strptime(start_raw, "%Y-%m-%dT%H:%M")
        end_dt = datetime.strptime(end_raw, "%Y-%m-%dT%H:%M")
        start_dt = start_dt.replace(tzinfo=IST)
        end_dt = end_dt.replace(tzinfo=IST)
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()
        election_data = {
            'id': f"ELEC{len(elections) + 1}-{datetime.now().strftime('%Y%m%d')}",
            'title': request.form.get('electionName'),
            'description': request.form.get('electionDescription'),
            'startDate': start_iso, 
            'endDate': end_iso,
            'status': 'Active',
            'published_results': False,
            'races': []
        }       
        race_indices = set()
        for key in form_data.keys():
            match = re.search(r'races\[(\d+)\]', key)
            if match:
                race_indices.add(match.group(1))
        for race_index in sorted(list(race_indices)):           
            race_entry = {
                'name': request.form.get(f'races[{race_index}][name]'),
                'type': request.form.get(f'races[{race_index}][type]'),
                'candidates': []
            }          
            candidate_map = {}
            for k, v in form_data.items():
                cand_match = re.search(r'races\[%s\]\[candidates\]\[(\d+)\]\[(name|party|photoUrl|region)\]' % race_index, k)
                if cand_match:
                    cand_index = cand_match.group(1)
                    field = cand_match.group(2)
                    if cand_index not in candidate_map:
                        candidate_map[cand_index] = {}
                    candidate_map[cand_index][field] = v[0] 
            for index, cand_data in candidate_map.items():
                region = cand_data.get('region', 'All Regions') 
                race_entry['candidates'].append({
                    'name': cand_data.get('name', 'N/A'),
                    'party': cand_data.get('party', 'N/A'),
                    'photoUrl': cand_data.get('photoUrl', '👤'),
                    'region': region 
                })
            election_data['races'].append(race_entry)
        elections.append(election_data)
        save_elections(elections)     
        flash(f'Election "{election_data["title"]}" successfully created and is now active!', 'success')
        return redirect(url_for('admin_dashboard'))
    default_regions = ['North District', 'South District', 'East District', 'West District', 'Central District', 'All Regions']  
    return render_template('truecast_createElections.html', default_regions=default_regions)

@app.route('/admin-logout')
def admin_logout():
    """Clears the admin session and redirects to admin login."""
    session.pop('admin_logged_in', None) 
    flash('You have been logged out of the Admin Panel.', 'success')
    return redirect(url_for('admin_login'))

@app.route('/admin/audit-and-publish/<string:election_id>', methods=['POST'])
@admin_required
def audit_and_publish(election_id):
    is_chain_valid, message = verify_vote_chain_integrity()
    if not is_chain_valid:
        print(f"CRITICAL AUDIT FAILURE for {election_id}: {message}")
        flash(f'AUDIT FAILED for {election_id}: Chain integrity check failed. Results CANNOT be published. Please run a full Audit from the dashboard.', 'error')
        return redirect(url_for('admin_dashboard'))
    elections = load_elections()
    found = False
    for i, election in enumerate(elections):
        if election.get('id') == election_id:
            if election.get('published_results', False):
                 flash(f'Results for "{election_id}" were already published.', 'warning')
                 return redirect(url_for('admin_dashboard'))
            elections[i]['published_results'] = True
            found = True
            break
    if found:
        save_elections(elections)
        flash(f'✅ Audit Passed & Results for "{election_id}" have been successfully PUBLISHED to the voters.', 'success')
    else:
        flash(f'Election ID {election_id} not found.', 'error')
    return redirect(url_for('admin_dashboard'))

@app.route('/results')
def results():
    votes_data = load_votes()
    elections = load_elections()    
    published_elections = [e for e in elections if e.get('published_results')]    
    target_election_id = request.args.get('election_id')
    display_election = None
    if target_election_id:
        display_election = get_election_by_id(target_election_id)    
    if display_election is None or not display_election.get('published_results'):
        display_election = next((e for e in reversed(published_elections)), None)
    if display_election is None:
        return render_template('truecast_results.html', 
                           results={}, 
                           election_title='No Published Results Available',
                           message='No election results have been certified and published by the administration.',
                           published_list=published_elections,
                           current_election_id=None)
    target_election_id = display_election['id']
    election_title = display_election.get('title', f"Results for {target_election_id}")
    results_tally = {}
    METADATA_KEYS = ['transactionHash', 'TransactionHash', 'electionId', 'ElectionId', 'timestamp','previous_hash', 
        'current_hash']
    target_election_id = display_election['id']
    election_title = display_election.get('title', f"Results for {target_election_id}")
    results_tally = {} 
    for vote_key, vote in votes_data.items():
        if isinstance(vote, dict) and vote.get('electionId') == target_election_id:
            for race_slug, candidate_slug in vote.items():
                if race_slug not in METADATA_KEYS: 
                    race_tally = results_tally.setdefault(race_slug, {})
                    race_tally[candidate_slug] = race_tally.get(candidate_slug, 0) + 1
    return render_template('truecast_results.html', 
                           results=results_tally, 
                           election_title=election_title,
                           message=None, 
                           published_list=published_elections, 
                           current_election_id=target_election_id)

@app.route('/admin/results')
@admin_required
def admin_live_results():
    voters_data = load_voters()
    votes_data = load_votes()
    elections = load_elections()
    active_election = get_active_election()    
    election_id = request.args.get('election_id')
    target_election = None    
    if election_id:
        target_election = get_election_by_id(election_id)    
    if not target_election:
        target_election = active_election        
    if not target_election and elections:
        target_election = next((e for e in reversed(elections) if e.get('status') == 'Ended'), None)
    if not target_election:
         return render_template('truecast_admin_results.html', 
                               results={}, 
                               election_title='No Elections Available',
                               is_active=False,
                               is_published=False,
                               election_id='N/A',
                               total_votes_in_election=0)
    target_election_id = target_election['id']
    election_title = target_election.get('title', f"Results for {target_election_id}")
    is_active = target_election.get('status') == 'Active'
    is_published = target_election.get('published_results', False)  
    results_tally = {}
    total_votes_in_election = 0    
    METADATA_KEYS_TO_EXCLUDE = [
        'transactionHash', 'TransactionHash', 
        'electionId', 'ElectionId', 
        'timestamp', 
        'previous_hash', 'current_hash' 
    ]
    for vote_key, vote in votes_data.items():
        if isinstance(vote, dict) and vote.get('electionId') == target_election_id: 
            total_votes_in_election += 1    
            for race_slug, candidate_slug in vote.items():
                if race_slug not in METADATA_KEYS_TO_EXCLUDE:
                    race_tally = results_tally.setdefault(race_slug, {})
                    race_tally[candidate_slug] = race_tally.get(candidate_slug, 0) + 1 
    total_eligible_voters = len(voters_data)
    if total_eligible_voters > 0:
        turnout_rate = round((total_votes_in_election / total_eligible_voters) * 100, 2)
    else:
        turnout_rate = 0.0
    overall_margin_percentage = 'N/A'
    tightest_margin_found = float('inf')
    tightest_race_name = ''
    for race_slug, candidates in results_tally.items():
        sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        total_votes_in_race = sum(candidates.values())        
        if len(sorted_candidates) >= 2 and total_votes_in_race > 0:
            winner_count = sorted_candidates[0][1]
            runner_up_count = sorted_candidates[1][1]           
            raw_margin = winner_count - runner_up_count
            margin_percentage = (raw_margin / total_votes_in_race) * 100          
            if margin_percentage < tightest_margin_found:
                tightest_margin_found = margin_percentage
                tightest_race_name = race_slug
    if tightest_race_name:
        overall_margin_percentage = round(tightest_margin_found, 2)
    available_regions = set()
    for race in target_election.get('races', []):
        for candidate in race.get('candidates', []):
            region = candidate.get('region')
            if region:
                available_regions.add(region)                
    sorted_available_regions = sorted(list(available_regions))    
    return render_template('truecast_admin_results.html', 
                           results=results_tally, 
                           election_title=election_title,
                           is_active=is_active,
                           is_published=is_published,
                           election_id=target_election_id,
                           total_votes_in_election=total_votes_in_election,
                           total_eligible_voters=total_eligible_voters,
                            turnout_rate=turnout_rate,
                            overall_margin_percentage=overall_margin_percentage,
                            available_regions=sorted_available_regions)

@app.route('/api/admin/get-chart-data/<string:election_id>')
@admin_required
def get_chart_data(election_id):
    voters_data = load_voters()
    votes_data = load_votes()    
    total_voters = len(voters_data)
    target_election = get_election_by_id(election_id)
    if not target_election:
        return jsonify({
            'turnout_data': {'labels': ['No Data'], 'data': [1]},
            'bar_chart_data': {'labels': ['No Races'], 'datasets': []}
        })   
    election_votes_cast = 0
    race_results = {}
    METADATA_KEYS = ['transactionHash', 'electionId', 'timestamp', 'previous_hash', 'current_hash']
    for vote_key, vote in votes_data.items():
        if isinstance(vote, dict) and vote.get('electionId') == election_id:
            election_votes_cast += 1
            for race_slug, candidate_slug in vote.items():
                if race_slug not in METADATA_KEYS:
                    race_display = race_slug.replace('-', ' ').title() 
                    race_tally = race_results.setdefault(race_display, {})
                    race_tally[candidate_slug] = race_tally.get(candidate_slug, 0) + 1
    not_voted_count = total_voters - election_votes_cast
    turnout_data = {
        'labels': ['Votes Cast', 'Eligible Voters (Not Voted)'],
        'data': [election_votes_cast, not_voted_count]
    }
    bar_chart_data = {'labels': [], 'datasets': []} 
    if race_results:
        top_race_slug = next(iter(race_results.keys()), None)
        if top_race_slug:
            race_data = race_results[top_race_slug]
            bar_chart_data['labels'] = [l.replace('-', ' ').title() for l in race_data.keys()] 
            bar_chart_data['datasets'].append({
                'label': top_race_slug,
                'data': list(race_data.values()),
            })
    return jsonify({
        'turnout_data': turnout_data,
        'bar_chart_data': bar_chart_data
    })

@app.route('/api/admin/decrypt-email/<string:voter_id>', methods=['POST'])
@admin_required
def decrypt_voter_email_api(voter_id):
    """API to securely decrypt and return a single voter's email for verification."""
    voters = load_voters()
    voter_info = voters.get(voter_id)
    if voter_info:
        encrypted_email = voter_info.get('email')
        if encrypted_email:
            decrypted_email = decrypt_data(encrypted_email)
            if decrypted_email != "ENCRYPTION_FAILED":
                return jsonify({'success': True, 'email': decrypted_email})
    return jsonify({'success': False, 'error': 'Voter not found or decryption failed.'}), 404

@app.route('/admin/check-integrity', methods=['POST'])
@admin_required
def check_integrity():
    """Runs the cryptographic chain verification audit and redirects."""    
    is_chain_valid, message = verify_vote_chain_integrity()
    if is_chain_valid:
        print(f"AUDIT SUCCESS: {message}")
        flash(f'Chain Verification Success: {message}', 'success')
    else:
        simple_message = "Chain integrity failed."
        if "Chain break detected" in message:
            simple_message = "Critical chain break detected. Vote history has been compromised."
        elif "Integrity break detected" in message:
            simple_message = "Data tampering detected. Vote content in one block is invalid."
        print(f"AUDIT FAILURE: {message}")
        flash(f'CRITICAL FAILURE: {simple_message} Please review the server log for technical details.', 'error')        
    return redirect(url_for('admin_dashboard'))

@app.route('/truecast_landing.html')
def truecast_landing():
    return redirect(url_for('home'))

@app.route('/help')
def help_page():
    return render_template('truecast_help.html')

@app.route('/about')
def about():
    return render_template('truecast_about.html')

@app.route('/accessibility')
def accessibility():
    return render_template('truecast_accessibility.html')

@app.route('/contactForm')
def contactForm():
    return render_template('truecast_contactForm.html')

@app.route('/privacypolicy')
def privacypolicy():
    return render_template('truecast_privacypolicy.html')

@app.route('/security')
def security():
    return render_template('truecast_security.html')

@app.route('/documentation')
def documentation():
    return render_template('truecast_documentation.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    if not chat_model:
        return jsonify({'response': "System Error: Chatbot service is unavailable."}), 500
    
    data = request.get_json()
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({'response': "Please enter a message."}), 400
        
    try:
        # 1. Initialize history if it doesn't exist
        if 'chat_history' not in session:
            session['chat_history'] = []

        # 2. Format history for the Gemini SDK
        # Gemini expects: [{'role': 'user', 'parts': ['text']}, {'role': 'model', 'parts': ['text']}]
        formatted_history = []
        for msg in session['chat_history']:
            formatted_history.append({
                'role': msg['role'],
                'parts': [msg['parts'][0]] # Accessing the first element of the parts list
            })

        # 3. Start chat and send message
        chat = chat_model.start_chat(history=formatted_history)        
        response = chat.send_message(user_message)
        
        # 4. Use .text safely
        bot_reply = response.text

        # 5. Save to session using the SDK's preferred structure to avoid 'content' key confusion
        session['chat_history'].append({'role': 'user', 'parts': [user_message]})
        session['chat_history'].append({'role': 'model', 'parts': [bot_reply]})
        
        session.modified = True 
        return jsonify({'response': bot_reply})

    except Exception as e:
        # This will now print the exact error to your terminal if it fails again
        print(f"Gemini Chat Error: {e}")
        return jsonify({'response': "I'm having trouble connecting right now. Please try again later."}), 500
if __name__ == "__main__":
    if not os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Error creating {JSON_FILE}: {e}")
    if not os.path.exists(VOTES_FILE):
        try:
            with open(VOTES_FILE, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Error creating {VOTES_FILE}: {e}")      
    if not os.path.exists(ELECTIONS_FILE):
        try:
            with open(ELECTIONS_FILE, 'w') as f:
                json.dump([], f)
        except Exception as e:
            print(f"Error creating {ELECTIONS_FILE}: {e}")
    app.run(debug=True)