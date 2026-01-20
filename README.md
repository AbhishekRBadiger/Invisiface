# InvisiFace  
**Enhancing Facial Recognition with Reconstruction Techniques**

InvisiFace is a computer vision project designed to reconstruct occluded facial regions—especially surgical mask–covered areas—using facial landmark detection and image inpainting. The system uses MediaPipe FaceMesh to detect facial structure and OpenCV’s inpainting to intelligently reconstruct missing facial regions.

The project is implemented in two modes:  
1. A standalone Python script for batch image processing  
2. A Streamlit-based web application for interactive use  

---

## Features

- Automatic face detection using MediaPipe FaceMesh  
- Smart rectangular mask region detection  
- Inpainting-based facial reconstruction  
- Optional FaceMesh visualization overlay  
- Streamlit web interface for easy testing  

---

## How It Works

1. Input image is loaded  
2. Face landmarks are detected using MediaPipe  
3. Mask region is estimated geometrically  
4. Region is reconstructed using OpenCV inpainting  
5. Optional FaceMesh overlay is applied  

---

## Project Structure

```
InvisiFace/
│
├── invisimask.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

```
pip install -r requirements.txt
```

---

## Run

### Standalone Script
```
python invisimask.py
```

### Streamlit App
```
streamlit run app.py
```

---

## Output

- 01_mask_removal_area.png  
- 02_inpainted_mask_removed.png  
- 03_facemesh_final.png  

---

## Notes

- Best with front-facing faces  
- Inpainting may create artifacts  
- Mask area is estimated  

---

## Applications

- Face recognition under occlusion  
- Forensics and surveillance  
- Research and learning projects  
