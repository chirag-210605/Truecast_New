import cv2
import numpy as np
import base64
from deepface import DeepFace 

def preprocess_image_data(base64_img_string):
    """
    Decodes a base64 string (from client-side JavaScript) into a NumPy array
    compatible with OpenCV/DeepFace.
    """
    try:
        if 'base64,' in base64_img_string:
            _, img_data = base64_img_string.split(',', 1)
        else:
            img_data = base64_img_string
        binary_data = base64.b64decode(img_data)
        np_array = np.frombuffer(binary_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Could not decode image data.")
            return None    
        return img
    except Exception as e:
        print(f"Error in preprocess_image_data: {e}")
        return None

def verify_face_match(registration_img_b64, login_img_b64):
    """
    Performs robust face verification using the DeepFace library.
    
    Args:
        registration_img_b64 (str): Base64 string of the registration photo.
        login_img_b64 (str): Base64 string of the login photo.
        
    Returns:
        tuple: (bool success, str message, float score/distance)
    """
    reg_img = preprocess_image_data(registration_img_b64)
    log_img = preprocess_image_data(login_img_b64)
    if reg_img is None or log_img is None:
        return False, "Failed to decode one or both images.", 0.0
    try:
        result = DeepFace.verify(
            img1_path = reg_img, 
            img2_path = log_img, 
            model_name = 'VGG-Face', 
            detector_backend = 'opencv',
            distance_metric = 'cosine',
            enforce_detection = True 
        )
        is_match = result['verified']
        distance = result['distance'] 
        threshold = result['threshold']    
        if is_match:
            return True, f"Face verified successfully (Distance: {distance:.4f} < Threshold: {threshold:.4f}).", round(distance, 4)
        else:
            return False, f"Verification failed. Distance: {distance:.4f} (Threshold: {threshold:.4f}).", round(distance, 4)
    except ValueError as e:
        if 'Face could not be detected' in str(e):
             return False, "Verification failed: A clear face could not be detected in one of the photos.", 1.0 
        return False, f"An unexpected verification error occurred: {e}", 1.0
    except Exception as e:
        return False, f"Internal verification error: {e}", 1.0