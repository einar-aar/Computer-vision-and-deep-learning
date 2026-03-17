from ultralytics import YOLO

def main():

    model = YOLO("runs/detect/pole_detection/YOLOv8s_960_ls/weights/best.pt")

    parameters = model.val(

        data = "roadpoles_v1/yolo_ls.yaml",
        imgsz = 960,
        device = 0
    )

    print("Recall: ", parameters.box.mr, " mAP@0.5:0.95: ", parameters.box.map, " Precision: ", parameters.box.mp, " mAP@50: ", parameters.box.map50)

if __name__ == "__main__":
    main()
