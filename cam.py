# Import required libraries
import torch  # PyTorch library for loading and running the YOLOv5 model
import cv2    # OpenCV library for video capture and drawing visualizations

# 1. Load the custom-trained YOLOv5 model
# 'custom' indicates we're using a custom model (not a built-in YOLOv5 model)
# 'path' points to the trained model weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='helmet_project/yolov5_fixed/weights/best.pt')

# Set the confidence threshold for predictions (detections below this will be ignored)
model.conf = 0.4

# 2. Open the video file
cap = cv2.VideoCapture('sources/helmet2.mp4')
if not cap.isOpened():
    print("Failed to open video.")
    exit()

# 3. Read and process each frame from the video
while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit the loop if no frame is returned (end of video)

    # 4. Perform inference using the loaded model
    results = model(frame)

    # 5. Extract predictions as a pandas DataFrame
    detections = results.pandas().xyxy[0]  # xyxy format: xmin, ymin, xmax, ymax

    # 6. Loop through each detection and visualize if it meets confidence threshold
    for _, row in detections.iterrows():
        label = row['name']          # Detected object class label (e.g., helmet, head, person)
        conf = row['confidence']     # Confidence score for the detection
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])  # Bounding box coordinates

        # Set the bounding box color based on class and confidence
        if label == 'helmet' and conf >= 0.5:
            color = (0, 255, 0)  # Green for helmet
        elif label == 'head' and conf >= 0.2:
            color = (0, 0, 255)  # Red for uncovered head
        elif label == 'person' and conf >= 0.4:
            color = (255, 0, 0)  # Cyan for person
        else:
            continue  # Skip detections with low confidence or unrelated classes

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 4)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 7. Display the processed frame in a window
    cv2.imshow("Helmet Detection AI", frame)

    # Press 'q' to exit the video playback early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Clean up: release video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()