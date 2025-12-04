from ultralytics import YOLO

def train_yolov8_segment():
    model = YOLO("yolov8s-seg.pt") 
    model.to("cuda")
    model.train(
        data="D:\Mon Hoc\Thi Giac May Tinh Thay Duy\Dataset_Segmentation\config.yaml", 
        epochs=30,
        imgsz=640,
        batch=8,
        workers=4,
        task="segment",
        mosaic=1.0,
        degrees=10,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        mixup=0.0,
        name="yolov8_segment_board",
        exist_ok=True
    )

if __name__ == "__main__":
    train_yolov8_segment()
