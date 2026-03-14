from ultralytics import YOLO

def main():

    model = YOLO(...)

    model.predict(

        source = "Road_poles_iPhone/images/Test/test",
        project = "Pole_detection",
        name = "yolov8s_iPhone_test",
        imgsz = 960,
        device = 0,
        save = True,
        save_txt = True
    )


if __name__ == "__main__":
    main()