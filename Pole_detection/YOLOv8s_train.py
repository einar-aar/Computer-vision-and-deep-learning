from ultralytics import YOLO

def main():

    model = YOLO("YOLOv8s.pt")

    model.train(

        data = "Combined/combined.yaml",

        imgsz = 960,
        epochs = 80,
        batch = 8,
        device = 0,
        workers = 4,
        project = "pole_detection",
        name = "YOLOv8s_960_combined"
    )

if __name__ == "__main__":
    main()