from ultralytics import YOLO

def main():

    model = YOLO("runs/detect/pole_detection/YOLOv8n_960_combined/weights/best.pt")

    model.predict(

        source = "Combined/images/test",
        project = "Pole_detection",
        name = "yolov8n_combined_test",
        imgsz = 960,
        device = 0,
        save = False,
        save_txt = True,
        save_conf = True
    )

if __name__ == "__main__":
    main()