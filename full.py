import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face = mp.solutions.face_mesh
face_mesh_model = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# --- UTILITIES ---

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found: " + path)
    return img

def apply_facemesh_overlay(image, landmarks_list):
    """Draws facemesh landmarks onto the image."""
    output_image = image.copy()
    for face in landmarks_list:
        for lm in face.landmark:
            x = int(lm.x * image.shape[1])
            y = int(lm.y * image.shape[0])
            cv2.circle(output_image, (x, y), 1, (0, 255, 0), -1)
    return output_image

def inpaint_region(image, mask):
    """Performs image inpainting."""
    # Use a large inpaint radius (e.g., 15) for large regions like a mask
    inpainted = cv2.inpaint(image, mask, 15, cv2.INPAINT_TELEA) 
    return inpainted

# --- NEW CORE LOGIC (Focused on Rectangular Mask for the Surgical Mask) ---

def get_mask_area(image):
    """
    1. Detects face landmarks to get a general face boundary.
    2. Creates a large rectangular mask covering the typical area of a surgical mask.
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = face_mesh_model.process(rgb)

    if not res.multi_face_landmarks:
        print("Warning: No face detected. Using default central mask area.")
        # Fallback to a fixed central area if no face is found
        x1, y1 = int(w * 0.25), int(h * 0.35)
        x2, y2 = int(w * 0.75), int(h * 0.80)
        landmarks_list = None
    else:
        face_landmarks = res.multi_face_landmarks[0]
        
        # Get the top of the nose (e.g., landmark 6, just below the eyes)
        lm_top = face_landmarks.landmark[6] 
        y_top = int(lm_top.y * h)
        
        # Get the bottom of the chin (e.g., landmark 152)
        lm_bottom = face_landmarks.landmark[152] 
        y_bottom = int(lm_bottom.y * h)
        
        # Get the left and right sides (e.g., landmarks 93 and 323, around the cheek)
        lm_left = face_landmarks.landmark[93]
        x_left = int(lm_left.x * w)
        
        lm_right = face_landmarks.landmark[323]
        x_right = int(lm_right.x * w)

        # Define the mask rectangle slightly expanded around the estimated mask area
        # Start the mask from just below the nose/eye line (approx 15% down from top)
        # End the mask near the chin
        y1 = int(y_top * 1.1)     # Start slightly below the detected nose/cheek area
        y2 = int(y_bottom * 0.95) # End slightly above the chin
        x1 = int(x_left * 0.85)   # Expand to the left
        x2 = int(x_right * 1.15)  # Expand to the right
        
        landmarks_list = res.multi_face_landmarks

    # Create the rectangular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    # The cv2.rectangle function draws a filled rectangle if thickness=-1
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # Apply a large feathering/blur to help with blending the mask boundaries
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    
    return mask, landmarks_list

# --- MAIN PIPELINE ---

def run_invisimask(path):
    print("Starting processing to remove the surgical mask...")
    original_img = load_image(path)
    
    # 1. GET RECTANGULAR MASK FOR THE SURGICAL MASK REGION
    mask, landmarks_list = get_mask_area(original_img)
    cv2.imwrite("01_mask_removal_area.png", mask)
    
    # 2. INPAINT THE MASKED AREA (TRIES TO GUESS SKIN TEXTURE)
    inpainted_img = inpaint_region(original_img, mask)
    cv2.imwrite("02_inpainted_mask_removed.png", inpainted_img)
    
    # 3. APPLY FACEMESH OVERLAY ON THE INPAINTED IMAGE (Optional)
    if landmarks_list:
        facemesh_img = apply_facemesh_overlay(inpainted_img, landmarks_list)
    else:
        facemesh_img = inpainted_img
        
    cv2.imwrite("03_facemesh_final.png", facemesh_img)
    
    print("✔ Processing complete! Check output images. Note: Inpainting a mask often results in artifacts.")
    return facemesh_img

# ----------------------------------------
# RUN
# ----------------------------------------
if __name__ == "__main__":
    # Ensure you are passing the path to the original image
    run_invisimask("faceimage5.jpeg")