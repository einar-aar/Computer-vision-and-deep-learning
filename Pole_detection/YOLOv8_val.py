from ultralytics import YOLO

def main():

    model = YOLO(...)

    parameters = model.val(

        data = "Road_poles_iPhone/yolo_iPhone",
        imgsz = 960,
        device = 0
    )

    print("Recall: ", parameters.box.mr, " mAP@0.5:0.95: ", parameters.box.map, " Precision: ", parameters.box.mp, " mAP@50: ", parameters.box.map50)

if __name__ == "__main__":
    main()