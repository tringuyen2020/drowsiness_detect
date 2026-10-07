import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from ultralytics import YOLO

# Load the models
cnn_model = load_model(os.path.join(os.path.dirname(__file__), "models", "drowsiness_detection_model.keras"))
yolo_model = YOLO("yolov8n.pt")
yolo_model.info()

# Setup camera
cam = cv2.VideoCapture(1)
count = 0
THRESHOLD_FRAMES = 15

if not cam.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    # Capture frame
    ret, frame = cam.read()
    
    if not ret:
        print("Error: Could not read frame")
        break

    # Process the frame
    # cv2.imshow("Frame", frame)

    height, width, _ = frame.shape
    results = yolo_model(frame, stream=True, verbose=False)
    face_detected = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            face_detected = True
            fx1, fy1, fx2, fy2 = map(int, box.xyxy[0].tolist())

            # draw face rectangle
            cv2.rectangle(frame, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 2)

            # eye region estimation
            face_height = fy2 - fy1
            eye_y1 = int(fy1 + (0.2 * face_height))
            eye_y2 = int(fy1 + (0.6 * face_height))

            # boundary check
            eye_y1, eye_y2 = max(0, eye_y1), min(height, eye_y2)
            fx1, fx2 = max(0, fx1), min(width, fx2)

            # eye region
            eye_region = frame[eye_y1:eye_y2, fx1:fx2]
            if eye_region.size == 0:
                continue

            # draw eye region
            cv2.rectangle(frame, (int(fx1), int(eye_y1)), (int(fx2), int(eye_y2)), (255, 0, 0), 2)

            # CNN model processing
            rgb_crop = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
            resized_crop = cv2.resize(rgb_crop, (64, 64))
            normalized_crop = resized_crop / 255.0

            # reshape to batch dimension
            input_tensor = np.expand_dims(normalized_crop, axis=0)

            # Make prediction
            prediction = cnn_model.predict(input_tensor, verbose=0)
            # print(f"Prediction: {prediction[0][0]}")

            if prediction[0][0] > 0.5:  # Adjust threshold as needed
                count += 1
                status = f"Closed: ({prediction[0][0]:.2f})"
                color = (0, 0, 255)
            else:
                count = max(0, count - 1)
                status = f"Open: ({prediction[0][0]:.2f})"
                color = (0, 255, 0)

            cv2.putText(frame, status, (int(fx1), int(fy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if not face_detected:
        count = max(0, count - 1)

    # show alert
    if count > THRESHOLD_FRAMES:
        cv2.putText(frame, "DROWSINESS ALERT", (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.putText(frame, f"Drowsiness Count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()