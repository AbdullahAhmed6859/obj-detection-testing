from ultralytics import YOLO


def main():
    print("Loading PyTorch model...")
    # Load the model you want to export
    model = YOLO("yolo26l.pt")

    print("Exporting to ONNX format...")
    # Export the model.
    # setting dynamic=True allows the model to accept varying image resolutions later
    export_path = model.export(format="onnx", dynamic=True)

    print(f"Success! Model exported to: {export_path}")


if __name__ == "__main__":
    main()
