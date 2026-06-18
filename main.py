import cv2
import time
from ultralytics import YOLO


def main():
    # Load the YOLOv26 nano model
    model = YOLO("yolo26x.pt")

    # Initialize the default webcam (index 0 is usually the built-in laptop camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Run inference directly on the current frame
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
        cv2.imshow("YOLOv26n Real-Time Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Gracefully release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
