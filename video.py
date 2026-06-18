import cv2
import time
from ultralytics import YOLO


def main():
    # Load the YOLOv26 nano model (corrected from yolov8n.pt)
    model = YOLO("yolo26x.mlpackage")

    # Pass the path to your video file instead of the webcam index '0'
    video_path = "videos/area-1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    prev_time = 0

    while True:
        success, frame = cap.read()

        # Break the loop smoothly when the video ends
        if not success:
            print("Video playback finished or failed to grab frame.")
            break

        # Run inference. Not specifying the 'classes' argument detects ALL classes.
        results = model(frame)

        # Plot the detections onto the frame
        annotated_frame = results[0].plot()

        # Calculate the framerate
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Draw the FPS counter on the top-left corner
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Render the view
        cv2.imshow("YOLOv8n Video Inference (All Classes)", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Gracefully release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
