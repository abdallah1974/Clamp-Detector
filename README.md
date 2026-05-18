# Clamp Detection and Counting using YOLOv11

This project implements an automated **clamp detection and counting system** using **YOLOv11m** and a custom annotated dataset created with Roboflow. The model is trained to accurately detect clamps in industrial video frames and perform real-time inference for localization and counting.

---

## Features

- Custom dataset annotation using Roboflow
- Training with YOLOv11m on Google Colab
- Real-time clamp detection
- Automatic clamp counting per frame
- Video inference with bounding box visualization
- Export of processed detection videos

---

## Tech Stack

- Python
- YOLOv11 (Ultralytics)
- OpenCV
- PyTorch
- Roboflow
- Google Colab

---

## Project Workflow

1. Extract frames from the input video
2. Annotate clamps using Roboflow
3. Export dataset in YOLO format
4. Train YOLOv11m model
5. Evaluate model performance
6. Run inference on unseen videos
7. Count detected clamps
8. Generate output detection video

---

## Project Structure

```bash
Clamp-Detection/
│
├── dataset/               # Roboflow exported dataset
├── models/
│   └── best.pt            # Trained YOLOv11 weights
│
├── main.py                # Inference and counting script
├── train.ipynb            # Google Colab training notebook
├── requirements.txt
├── test_video.mp4
├── output_clamps.mp4
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd Clamp-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---


## Run Inference

best trained weights inside:

```bash
models/best.pt
```

Then run:

```bash
python main.py
```

---

## Output

The script will:

- Detect clamps frame-by-frame
- Draw bounding boxes
- Count clamps in each frame
- Generate:

```bash
output_clamps.mp4
```

---

## Sample Detection

The model performs real-time clamp detection and counting for industrial inspection and monitoring applications.

---

## Future Improvements

- Multi-object tracking for persistent clamp IDs
- Unique clamp counting across video frames
- Deployment with Streamlit
- Edge deployment on NVIDIA Jetson

---

## Author

**Abdallah Hassan**

