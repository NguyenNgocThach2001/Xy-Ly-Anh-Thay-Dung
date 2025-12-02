import cv2
import numpy as np
from ultralytics import YOLO

def extract_quad_from_mask(mask):
    mask = (mask.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2)
    return None

def run_inference_webcam(model_path="runs/segment/yolov8_segment_board/weights/best.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture("http://192.168.1.97:4747/video")

    if not cap.isOpened():
        print("Không mở được webcam.")
        return

    print("Đang chạy webcam. Nhấn ESC để thoát...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.4, verbose=False)
        for r in results:
            if r.masks is not None:
                for mask in r.masks.data:
                    quad = extract_quad_from_mask(mask)
                    if quad is not None:
                        cv2.polylines(frame, [quad.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [quad.astype(int)], color=(0, 255, 0))
                        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        cv2.imshow("YOLOv8 Segment → Quadrilateral", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference_webcam()
