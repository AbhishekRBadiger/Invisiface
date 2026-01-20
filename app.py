import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

# --- 1. COMPUTER VISION CORE FUNCTIONS ---
# Initialize MediaPipe FaceMesh
mp_face = mp.solutions.face_mesh
face_mesh_model = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def inpaint_region(image, mask):
    """Performs image inpainting."""
    # Use a large inpaint radius (e.g., 15) for large regions like a mask
    # cv2.INPAINT_TELEA is generally better for guessing structure
    inpainted = cv2.inpaint(image, mask, 15, cv2.INPAINT_TELEA) 
    return inpainted

def get_mask_area(image_bgr):
    """
    1. Detects face landmarks to get a general face boundary.
    2. Creates a large rectangular mask covering the typical area of a surgical mask.
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh_model.process(rgb)
    landmarks_list = None
    
    if not res.multi_face_landmarks:
        # Fallback to a fixed central area if no face is found
        st.warning("No face detected. Using default central mask area.")
        y1, y2 = int(h * 0.35), int(h * 0.80)
        x1, x2 = int(w * 0.25), int(w * 0.75)
    else:
        face_landmarks = res.multi_face_landmarks[0]
        landmarks_list = res.multi_face_landmarks
        
        # Get reference landmarks for the mask boundary
        # Top of the mask area (just below nose/eyes, e.g., landmark 6)
        lm_top = face_landmarks.landmark[6] 
        y_top = int(lm_top.y * h)
        
        # Bottom of the chin (e.g., landmark 152)
        lm_bottom = face_landmarks.landmark[152] 
        y_bottom = int(lm_bottom.y * h)
        
        # Left and right sides (e.g., landmarks 93 and 323, around the cheek)
        lm_left = face_landmarks.landmark[93]
        x_left = int(lm_left.x * w)
        
        lm_right = face_landmarks.landmark[323]
        x_right = int(lm_right.x * w)

        # Define the mask rectangle with padding
        y1 = int(y_top * 1.1)     
        y2 = int(y_bottom * 0.95)
        x1 = int(x_left * 0.90)   
        x2 = int(x_right * 1.10)  

    # Create the rectangular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # Apply a large feathering/blur to help with blending the mask boundaries
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    
    return mask, landmarks_list

def apply_facemesh_overlay(image_bgr, landmarks_list):
    """Draws facemesh landmarks onto the image."""
    output_image = image_bgr.copy()
    for face in landmarks_list:
        for lm in face.landmark:
            x = int(lm.x * image_bgr.shape[1])
            y = int(lm.y * image_bgr.shape[0])
            cv2.circle(output_image, (x, y), 1, (0, 255, 0), -1)
    return output_image

def process_invisimask(image_bgr):
    """Main function to run the mask removal pipeline."""
    # 1. Get Mask Area
    mask, landmarks_list = get_mask_area(image_bgr)
    
    # 2. Inpaint the Masked Area
    inpainted_img = inpaint_region(image_bgr, mask)
    
    # 3. Apply FaceMesh Overlay (Optional, but useful for visualization)
    if landmarks_list:
        facemesh_img = apply_facemesh_overlay(inpainted_img, landmarks_list)
    else:
        facemesh_img = inpainted_img
        
    return facemesh_img

# --- 2. STREAMLIT GUI LAYOUT ---

def main():
    st.set_page_config(page_title="InvisiFace.AI", layout="centered")

    # Header and Title
    st.title("InvisiFace.AI")
    st.markdown("---")
    st.header("Upload Image, Drag & Drop, or Capture from Webcam")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Drag & Drop Image Here or Click to Upload",
        type=['png', 'jpg', 'jpeg']
    )

    # --- PROCESSING LOGIC ---
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded Image:")
        
        # Read the image bytes and convert to OpenCV format (BGR)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img_bgr = cv2.imdecode(file_bytes, 1)
        
        # Convert BGR to RGB for Streamlit display
        original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(original_img_rgb, caption=f"Uploaded File: {uploaded_file.name}", use_column_width=True)

        st.markdown("---")
        
        # Process Button
        if st.button("Process Image", type="primary"):
            with st.spinner('Processing image...'):
                try:
                    # Run the computer vision pipeline
                    processed_img_bgr = process_invisimask(original_img_bgr)
                    
                    # Convert back to RGB for Streamlit display
                    processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)
                    
                    st.success("Processing Complete!")
                    st.subheader("Processed Image:")
                    st.image(processed_img_rgb, caption="Mask Removed and FaceMesh Applied", use_column_width=True)

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
    else:
        st.info("Awaiting image upload...")

if __name__ == "__main__":
    main()