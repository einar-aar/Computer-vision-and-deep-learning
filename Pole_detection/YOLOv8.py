from ultralytics import YOLO

def main():

    # Choose what pretrained model and save it in model variable
    model = YOLO("YOLOv8s.pt")

    # The model variable har a train function for training
    model.train(

        data = "Road_poles_iPhone/iPhone.yaml",

        imgsz = 960,
        epochs = 5,
        batch = 8,
        device = 0,
        workers = 4,
        project = "pole_detection",
        name = "YOLOv8s_960_iPhone"
    )

if __name__ == "__main__":
    main()