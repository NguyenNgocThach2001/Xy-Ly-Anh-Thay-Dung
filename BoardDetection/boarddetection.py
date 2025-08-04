import cv2
import numpy as np

def is_rectangle(pts, angle_thresh=15):
    """Kiểm tra tứ giác có 4 góc gần 90°"""
    def angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    pts = order_points(np.array(pts, dtype="float32"))
    angles = [
        angle(pts[3], pts[0], pts[1]),
        angle(pts[0], pts[1], pts[2]),
        angle(pts[1], pts[2], pts[3]),
        angle(pts[2], pts[3], pts[0])
    ]
    return all(abs(a - 90) < angle_thresh for a in angles)

def order_points(pts):
    """Sắp xếp điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# === Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Smooth (Gaussian)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Detect edges
    edges = cv2.Canny(thresh, 50, 150)

    # Step 5: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        # Step 6: Approximate polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # Step 7: Check if rectangle-like
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            if is_rectangle(pts):
                if area > max_area:
                    best_rect = pts
                    max_area = area

    display = frame.copy()

    if best_rect is not None:
        ordered = order_points(best_rect)
        cv2.polylines(display, [ordered.astype(int)], True, (0, 255, 0), 2)
        for pt in ordered:
            cv2.circle(display, tuple(pt.astype(int)), 5, (255, 0, 0), -1)

        # Warp
        (tl, tr, br, bl) = ordered
        width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        dst = np.array([[0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(frame, M, (int(width), int(height)))
        cv2.imshow("Warped", warped)

    # Hiển thị
    cv2.imshow("Detected Rectangle", display)
    cv2.imshow("Canny", edges)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
