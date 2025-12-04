import os, shutil, random, math
from pathlib import Path

# ==== CHỈNH ĐƯỜNG DẪN CHO PHÙ HỢP ====
ROOT = Path("Dataset_Detection/piece/Dataset_combined")
SRC = ROOT / "train"        # train/ hiện có (ảnh + nhãn)
OLD = ROOT / "train_old"    # sẽ đổi tên train/ -> train_old/

# ==== TỈ LỆ CHUẨN ====
PCT_TRAIN = 0.70  # 70%
PCT_VAL   = 0.20  # 20%
PCT_TEST  = 0.10  # 10%

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def reset_split_dir(root: Path):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)

def paired_label(img_path: Path, labels_root: Path) -> Path:
    return labels_root / (img_path.stem + ".txt")

def compute_counts(n):
    # làm tròn để tổng = n
    n_train = int(round(n * PCT_TRAIN))
    n_val   = int(round(n * PCT_VAL))
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test

def main():
    assert SRC.exists(), f"Không thấy thư mục {SRC}"

    # 1) Đổi tên train/ hiện có -> train_old/ (nguồn)
    if OLD.exists():
        raise SystemExit(f"{OLD} đã tồn tại, hãy xoá/đổi tên rồi chạy lại.")
    shutil.move(str(SRC), str(OLD))

    # 2) Tạo lại 3 split: train/, val/, test/
    for d in [ROOT / "train", ROOT / "val", ROOT / "test"]:
        reset_split_dir(d)

    # 3) Gom ảnh có nhãn hợp lệ
    imgs_root = OLD / "images"
    labels_root = OLD / "labels"
    if not imgs_root.exists() or not labels_root.exists():
        raise SystemExit("Nguồn không có images/ hoặc labels/ trong train_old/")

    # xoá cache cũ nếu có
    for cache in [OLD / "labels.cache", ROOT / "labels.cache"]:
        if cache.exists():
            cache.unlink()

    imgs = [p for p in imgs_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    imgs = [p for p in imgs if paired_label(p, labels_root).exists()]

    random.seed(42)
    random.shuffle(imgs)

    n = len(imgs)
    n_train, n_val, n_test = compute_counts(n)

    train_imgs = imgs[:n_train]
    val_imgs   = imgs[n_train:n_train + n_val]
    test_imgs  = imgs[n_train + n_val:]

    def move_pair(img_list, out_root: Path):
        for img_path in img_list:
            lbl_path = paired_label(img_path, labels_root)
            rel = img_path.relative_to(imgs_root)  # giữ cấu trúc con nếu có
            (out_root / "images" / rel.parent).mkdir(parents=True, exist_ok=True)
            (out_root / "labels" / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_path), str(out_root / "images" / rel))
            shutil.move(str(lbl_path), str(out_root / "labels" / rel.with_suffix(".txt")))

    move_pair(train_imgs, ROOT / "train")
    move_pair(val_imgs,   ROOT / "val")
    move_pair(test_imgs,  ROOT / "test")

    print(f"Done. Total: {n} | train: {len(train_imgs)} | val: {len(val_imgs)} | test: {len(test_imgs)}")
    print(f"Tạo thư mục: {ROOT / 'train'}, {ROOT / 'val'}, {ROOT / 'test'}")
    print(f"Nguồn còn lại: {OLD} (có thể xoá sau khi kiểm tra)")

if __name__ == "__main__":
    main()
