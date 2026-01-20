InvisiFace

Enhancing Facial Recognition with Reconstruction Techniques

InvisiFace is a computer vision project designed to reconstruct occluded facial regions—especially surgical mask–covered areas—using facial landmark detection and image inpainting. The system uses MediaPipe FaceMesh to detect facial structure and OpenCV’s inpainting to intelligently reconstruct missing facial regions.

The project is implemented in two modes:

A standalone Python script for batch image processing

A Streamlit-based web application for interactive use

🚀 Features

Automatic face detection using MediaPipe FaceMesh

Smart rectangular mask region detection

Inpainting-based facial reconstruction

Optional FaceMesh visualization overlay  

Streamlit web interface for easy testing

Works on images with or without detected faces

🧠 How It Works

Input image is loaded

Face landmarks are detected using MediaPipe

A rectangular region covering the typical mask area is generated

The masked region is reconstructed using OpenCV inpainting

Optional FaceMesh overlay is applied

Final output is displayed or saved

📂 Project Structure
InvisiFace/
│
├── invisimask.py         Standalone script
├── app.py                Streamlit web app
├── requirements.txt
└── README.md

⚙️ Installation
1. Clone the Repository
git clone https://github.com/your-username/invisiface.git
cd invisiface

2. Install Dependencies
pip install -r requirements.txt

▶️ Run the Programs
Program 1: Standalone Script
python invisimask.py


Make sure to update this line in the code:

run_invisimask("faceimage5.jpeg")

Program 2: Streamlit Web App
streamlit run app.py


Then open the link shown in terminal.

🖼 Output Files (Program 1)

01_mask_removal_area.png – Mask region

02_inpainted_mask_removed.png – Reconstructed face

03_facemesh_final.png – Final output with landmarks
