import cv2
import numpy as np
import os
import random

"""
yolo classify train data=dataset model=yolo11n-cls.pt epochs=50 imgsz=128

######################################################## V2 ########################################################
from ultralytics import YOLO
model = YOLO("runs/classify/train/weights/best.pt")

actual = left_hand   # 你切出來這張 A♦(BGR,不要轉 RGB!)
result = model(actual, verbose=False)[0]
print(result.names[result.probs.top1], float(result.probs.top1conf))
####################################################################################################################


# 載入一次(放在類別啟動時,別每次重載)
from ultralytics import YOLO
model = YOLO("runs/classify/train/weights/best.pt")

def predict_card(card_roi_bgr):
    # YOLO 吃 RGB,OpenCV 是 BGR,要轉
    rgb = cv2.cvtColor(card_roi_bgr, cv2.COLOR_BGR2RGB)
    result = model(rgb, verbose=False)[0]
    # 取機率最高的類別
    cls_id = result.probs.top1
    name = result.names[cls_id]          # 'spade_13'
    conf = float(result.probs.top1conf)  # 信心
    return name, conf


# 接進去 ImageManager (GameManager in the future)
y1, y2, x1, x2 = i.px["public_card_1"]
roi = i.img[y1:y2, x1:x2]
name, conf = predict_card(roi)
print(name, round(conf, 3))
"""


"""
50 epochs completed in 1.739 hours.
Optimizer stripped from /Users/tp_mini/Desktop/IB/natural8/runs/classify/train-2/weights/last.pt, 3.3MB
Optimizer stripped from /Users/tp_mini/Desktop/IB/natural8/runs/classify/train-2/weights/best.pt, 3.3MB

Validating /Users/tp_mini/Desktop/IB/natural8/runs/classify/train-2/weights/best.pt...
Ultralytics 8.4.80 🚀 Python-3.12.11 torch-2.12.1 CPU (Apple M2 Max)
YOLO11n-cls summary (fused): 47 layers, 1,592,636 parameters, 0 gradients, 3.2 GFLOPs
train: /Users/tp_mini/Desktop/IB/natural8/dataset/train... found 6292 images in 52 classes ✅ 
val: /Users/tp_mini/Desktop/IB/natural8/dataset/val... found 1560 images in 52 classes ✅ 
test: None...
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 49/49 4.9it/s 10.0s
                   all      0.988          1
Speed: 0.0ms preprocess, 5.5ms inference, 0.0ms loss, 0.0ms postprocess per image
Results saved to /Users/tp_mini/Desktop/IB/natural8/runs/classify/train-2
💡 Learn more at https://docs.ultralytics.com/modes/train
((yolo_venv) ) tp_mini@lizongyous-Mac-Studio natural8 % 

"""
class TrainDataGenerator:
    def __init__(self, src_file_path='data_card/',
                 background_path='data/green_bg_color.png',
                 n_per_card=150, val_ratio=0.2):
        self.src_file_path = src_file_path
        self.background_png_path = background_path
        self.n_per_card = n_per_card
        self.val_ratio = val_ratio                 # 多少比例分去驗證集
        self.TABLE_GREEN = self._get_table_green()

    def _get_table_green(self):
        bg = cv2.imread(self.background_png_path)
        if bg is None:
            raise FileNotFoundError(f"讀不到背景圖 {self.background_png_path}")
        b, g, r = bg[50, 50]
        return (int(b), int(g), int(r))

    # ---------- 五種增強 ----------
    def aug_shift(self, card, max_shift_ratio=0.1):
        h, w = card.shape[:2]
        max_dx = max(1, int(w * max_shift_ratio))
        max_dy = max(1, int(h * max_shift_ratio))
        dx = np.random.randint(-max_dx, max_dx + 1)
        dy = np.random.randint(-max_dy, max_dy + 1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(card, M, (w, h), borderValue=self.TABLE_GREEN)

    def aug_border(self, card, max_border_ratio=0.12):
        h, w = card.shape[:2]
        mb = max(1, int(min(h, w) * max_border_ratio))
        top, bottom = np.random.randint(0, mb + 1), np.random.randint(0, mb + 1)
        left, right = np.random.randint(0, mb + 1), np.random.randint(0, mb + 1)
        bordered = cv2.copyMakeBorder(card, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=self.TABLE_GREEN)
        return cv2.resize(bordered, (w, h))

    def aug_crop(self, card, max_crop_ratio=0.1):
        h, w = card.shape[:2]
        cx = int(w * np.random.uniform(0, max_crop_ratio))
        cy = int(h * np.random.uniform(0, max_crop_ratio))
        cropped = card[cy:h - cy, cx:w - cx]
        if cropped.size == 0:
            return card
        return cv2.resize(cropped, (w, h))

    def aug_brightness(self, card):
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.randint(-30, 31)
        return cv2.convertScaleAbs(card, alpha=alpha, beta=beta)

    def aug_blur(self, card):
        k = int(np.random.choice([3, 5]))
        return cv2.GaussianBlur(card, (k, k), 0)

    # ---------- 隨機複合 ----------
    def augment(self, card):
        out = card.copy()
        if np.random.rand() < 0.7:
            out = self.aug_shift(out)
        if np.random.rand() < 0.5:
            out = self.aug_border(out)
        if np.random.rand() < 0.3:
            out = self.aug_crop(out)
        if np.random.rand() < 0.6:
            out = self.aug_brightness(out)
        if np.random.rand() < 0.3:
            out = self.aug_blur(out)
        return out

    # ---------- 生成完整資料集(含 train/val 分類資料夾) ----------
    def run_generator(self, dataset_root='dataset'):
        card_files = [f for f in os.listdir(self.src_file_path)
                      if f.lower().endswith('.png')]
        print(f"找到 {len(card_files)} 張牌卡")

        for fname in card_files:
            name = fname[:-4]                          # 'spade_13'
            card = cv2.imread(os.path.join(self.src_file_path, fname))
            if card is None:
                print(f"讀不到 {fname},跳過")
                continue

            # 為這個類別建 train 和 val 資料夾
            train_dir = os.path.join(dataset_root, "train", name)
            val_dir = os.path.join(dataset_root, "val", name)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            n_val = int(self.n_per_card * self.val_ratio)
            for i in range(self.n_per_card):
                img_aug = self.augment(card)
                # 前 n_val 張放 val,其餘放 train
                out_dir = val_dir if i < n_val else train_dir
                cv2.imwrite(os.path.join(out_dir, f"{name}_{i:04d}.png"), img_aug)

            # 乾淨原圖也放一張到 train(讓模型也看過無偏差版本)
            cv2.imwrite(os.path.join(train_dir, f"{name}_orig.png"), card)

        print(f"完成。每類 {self.n_per_card} 張(其中 ~{int(self.n_per_card*self.val_ratio)} 張驗證)")


if __name__ == "__main__":
    gen = TrainDataGenerator(n_per_card=150)   # 每類 150 張
    gen.run_generator("dataset")