import numpy as np

# MediaPipe face mesh landmark indices for eyes and mouth
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 308, 13, 14, 87, 317] # Outer/inner lips vertical/horizontal landmarks

def calculate_ear(landmarks, eye_indices, image_width, image_height):
    """Computes the Eye Aspect Ratio (EAR) given face mesh landmarks."""
    coords = np.array([[landmarks[i].x * image_width, landmarks[i].y * image_height] for i in eye_indices])
    
    # Vertical distances
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    
    # Horizontal distance
    h = np.linalg.norm(coords[0] - coords[3])
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

def calculate_mar(landmarks, mouth_indices, image_width, image_height):
    """Computes the Mouth Aspect Ratio (MAR) to detect yawning."""
    coords = np.array([[landmarks[i].x * image_width, landmarks[i].y * image_height] for i in mouth_indices])
    
    v = np.linalg.norm(coords[2] - coords[3]) # Vertical inner lip distance
    h = np.linalg.norm(coords[0] - coords[1]) # Horizontal lip width
    
    mar = v / h
    return mar
