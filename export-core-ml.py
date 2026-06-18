from ultralytics import YOLO

names = ['glove', 'hand', 'safety helmet', 'reflective jacket']

def main():
    print("Loading PyTorch model...")
    model = YOLO("yoloe-26x-seg.pt")
    model.set_classes(names, model.get_text_pe(names))

    print("Exporting to ONNX format...")
    export_path = model.export(
        format="coreml",
        nms=True
    )

    print(f"Success! Model exported to: {export_path}")

if __name__ == "__main__":
    main()