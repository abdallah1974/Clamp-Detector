from ultralytics import YOLO
import cv2


# ============================================
# Paths
# ============================================
MODEL_PATH = "model/best.pt"        # trained model
VIDEO_PATH = "Test_Video.mp4"       # original video
OUTPUT_PATH = "output_clamps.mp4" 

# ============================================
# Load model
# ============================================
model = YOLO(MODEL_PATH)


# ============================================
# Open video
# ============================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# ============================================
# Video properties
# ============================================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 1 else 30


# ============================================
# Output writer
# ============================================
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# ============================================
# Display window
# ============================================
WINDOW_NAME = "Clamp Detection"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)


# ============================================
# Detection loop
# ============================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Faster inference
    results = model(
        frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # Count clamps
    clamp_count = len(result.boxes)

    # Draw detections
    annotated_frame = result.plot()

    # Add count
    cv2.putText(
        annotated_frame,
        f"Clamp Count: {clamp_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Save original size
    writer.write(annotated_frame)

    # Resize ONLY for display → smoother preview
    display_frame = cv2.resize(
        annotated_frame,
        (960, 540)
    )

    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ============================================
# Cleanup
# ============================================
cap.release()
writer.release()
cv2.destroyAllWindows()

print("Done.")