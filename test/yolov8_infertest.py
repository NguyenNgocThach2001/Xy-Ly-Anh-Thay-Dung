import os
import re

# === CẤU HÌNH ===
runs_dir = "runs/detect"
k_folds = 5

best_fold = None
best_map = -1.0

def parse_map_from_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"all.+?mAP50-95\):.+?([\d\.]+)", line)
            if match:
                return float(match.group(1))
    return -1.0

# === Tìm fold tốt nhất ===
for fold in range(1, k_folds + 1):
    result_path = os.path.join(runs_dir, f"xiangqi_fold{fold}", "results.txt")
    if os.path.exists(result_path):
        mAP = parse_map_from_results(result_path)
        print(f"📊 Fold {fold}: mAP50-95 = {mAP:.4f}")
        if mAP > best_map:
            best_map = mAP
            best_fold = fold
    else:
        print(f"❌ Không tìm thấy: {result_path}")

# === Kết quả ===
if best_fold is not None:
    print(f"\n🏆 Fold tốt nhất: Fold {best_fold} với mAP50-95 = {best_map:.4f}")
else:
    print("❌ Không tìm thấy kết quả hợp lệ.")
