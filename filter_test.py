import cv2
import numpy as np
import time

# --------- Utility ----------
def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def from_lab(lab):
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def luminance_bgr(img):
    # Rec.709 luma
    return 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

def put_label(im, text, y=28):
    cv2.rectangle(im, (0,0), (im.shape[1], y+10), (0,0,0), -1)
    cv2.putText(im, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

# --------- Filters ----------
def filt_bilateral_gamma(img, d=9, sigma_color=75, sigma_space=75, gamma=0.9):
    # Bilateral làm dịu highlight mà giữ biên; gamma<1 hơi nén vùng sáng
    base = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    base_f = np.clip(base.astype(np.float32)/255.0, 0, 1)
    out = np.power(base_f, gamma)
    return (out*255).astype(np.uint8)

def filt_edgepreserve_blend(img, sigma_s=60, sigma_r=0.4, blend=0.7):
    # Lọc bảo toàn biên rồi blend về ảnh gốc để bớt plastic
    ep = cv2.edgePreservingFilter(img, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    out = cv2.addWeighted(ep, blend, img, 1.0-blend, 0)
    return out

def filt_reinhard(img, intensity=0.0, light_adapt=1.0, color_adapt=0.0):
    # Reinhard tone mapping: nén highlight “dịu” không cần HDR input
    imf = img.astype(np.float32)/255.0
    tonemap = cv2.createTonemapReinhard(gamma=1.0, intensity=intensity,
                                        light_adapt=light_adapt, color_adapt=color_adapt)
    tm = tonemap.process(imf)  # returns float32
    tm = np.clip(tm, 0, 1)
    return (tm*255).astype(np.uint8)

def filt_filmic(img, strength=1.0):
    # Filmic curve (Hable) trên kênh L để roll-off highlight
    A, B, C, D, E, F = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30
    def hable(x):
        return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F
    lab = to_lab(img)
    L, A_, B_ = cv2.split(lab)
    Lf = L.astype(np.float32)/255.0 * 2.0  # scale lên chút để curve “ăn”
    mapped = hable(Lf)
    white = hable(11.2)  # white point
    mapped = np.clip(mapped/white, 0, 1)
    # blend với L gốc để đỡ quá tay
    outL = (strength*mapped + (1-strength)*(L.astype(np.float32)/255.0))
    outL = (np.clip(outL,0,1)*255).astype(np.uint8)
    return from_lab(cv2.merge([outL, A_, B_]))

# --------- Main loop ----------
def main():
    cap = cv2.VideoCapture(0)  # đổi 1/2 nếu cần
    if not cap.isOpened():
        print("❌ Không mở được webcam"); return

    win = "Webcam: Original | Filtered"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    mode = 1  # mặc định bật filter nhẹ
    names = {
        0: "OFF",
        1: "Bilateral + Gamma",
        2: "EdgePreserve + Blend",
        3: "Reinhard Tonemap",
        4: "Filmic Curve"
    }

    print("[INFO] Nhấn phím 0..4 để đổi filter, q/ESC để thoát.")
    t0 = time.time(); frames = 0; fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break

        # chọn filter
        if mode == 0:
            filtered = frame
        elif mode == 1:
            filtered = filt_bilateral_gamma(frame, d=9, sigma_color=60, sigma_space=60, gamma=0.9)
        elif mode == 2:
            filtered = filt_edgepreserve_blend(frame, sigma_s=60, sigma_r=0.35, blend=0.65)
        elif mode == 3:
            filtered = filt_reinhard(frame, intensity=0.0, light_adapt=1.0, color_adapt=0.0)
        elif mode == 4:
            filtered = filt_filmic(frame, strength=0.75)
        else:
            filtered = frame

        # FPS
        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames/dt
            t0 = time.time(); frames = 0

        left, right = frame.copy(), filtered.copy()
        put_label(left,  f"ORIGINAL (FPS~{fps:.1f})")
        put_label(right, f"{names.get(mode,'?')}")

        # ghép hai ảnh cùng chiều cao
        h = min(left.shape[0], right.shape[0])
        left  = cv2.resize(left,  (int(left.shape[1]*h/left.shape[0]),  h))
        right = cv2.resize(right, (int(right.shape[1]*h/right.shape[0]), h))
        combo = cv2.hconcat([left, right])

        cv2.imshow(win, combo)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        elif k in (ord('0'), ord('1'), ord('2'), ord('3'), ord('4')):
            mode = int(chr(k))
            print(f"[MODE] -> {mode}: {names[mode]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
