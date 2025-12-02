# Create a clean-color version (no hatching) of the previous chart
import matplotlib.pyplot as plt
import numpy as np

pieces = ["tướng", "hậu", "xe", "mã", "tượng", "tốt"]

# Keep the same equal distribution between colors and same ratios
others = 50
tot = others * 2.5   # 125
tuong = tot / 5      # 25

red_counts = [tuong, others, others, others, others, tot]
black_counts = [tuong, others, others, others, others, tot]

x = np.arange(len(pieces))
width = 0.38

plt.figure(figsize=(12, 6))
bars_red = plt.bar(x - width/2, red_counts, width, color="b", label="Đỏ")
bars_black = plt.bar(x + width/2, black_counts, width, color="g", label="Đen")

# Annotate
for bars in [bars_red, bars_black]:
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=10)

plt.xticks(x, pieces)
plt.ylabel("Số nhãn (labels)")
plt.xlabel("Loại quân cờ")
total_labels = sum(red_counts) + sum(black_counts)
plt.title(f"Phân bố nhãn")
plt.legend()
plt.ylim(0, max(red_counts + black_counts) + 25)
plt.tight_layout()

plt.show()
