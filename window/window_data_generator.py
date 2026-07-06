import cv2
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

"""
source ~/yolo_venv/bin/activate
yolo detect train data=window_data.yaml model=yolo11n.pt epochs=50 imgsz=640

detect train(不是 classify train)
model=yolo11n.pt(不是 -cls.pt)
data=window_data.yaml(指向 yaml 檔,不是資料夾)
imgsz=640(偵測要看清邊界,用大一點)
"""


DATA_FOLDER = 'window_dataset'
TRAIN_IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'images', 'train')
VALIDATE_IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'images', 'validate')
TRAIN_LABEL_FOLDER = os.path.join(DATA_FOLDER, 'labels', 'train')
VALIDATE_LABEL_FOLDER = os.path.join(DATA_FOLDER, 'labels', 'validate')

def synthesize_one(
    bg_fname=None,
    gameplay_window_fname=None,
    scale_min=0.4,
    scale_max=0.9,
):
    DATA_FOLDER = "window_data_raw"
    # 用傳進來的檔名,沒傳就用預設
    bg_path = bg_fname or f"{DATA_FOLDER}/wallpaper_example.png"
    gp_path = gameplay_window_fname or f"{DATA_FOLDER}/gameplay1.png"

    img = cv2.imread(bg_path)
    img_gameplay_window = cv2.imread(gp_path)

    canvas = img.copy()
    bg_h, bg_w = img.shape[0:2]

    # ---- 隨機縮放遊戲視窗 ----
    scale = random.uniform(scale_min, scale_max)
    orig_h, orig_w = img_gameplay_window.shape[0:2]
    gp_w = int(orig_w * scale)
    gp_h = int(orig_h * scale)
    # 確保縮放後不超過背景
    gp_w = min(gp_w, bg_w)
    gp_h = min(gp_h, bg_h)
    gp_resized = cv2.resize(img_gameplay_window, (gp_w, gp_h))

    # ---- 隨機位置 ----
    start_w = random.randint(0, bg_w - gp_w)
    start_h = random.randint(0, bg_h - gp_h)
    canvas[start_h:start_h+gp_h, start_w:start_w+gp_w] = gp_resized

    # ---- YOLO label(用縮放後的 gp_w, gp_h)----
    cw = (start_w + gp_w / 2) / bg_w
    ch = (start_h + gp_h / 2) / bg_h
    bw = gp_w / bg_w
    bh = gp_h / bg_h
    label = (0, cw, ch, bw, bh)

    return canvas, label

def init_dataset_folders():
    """
    window_detection_dataset/
        images/
            train/
                1.png
                ...
            validate/
                100.png
                ...
        labels/
            train/
                1.txt
            validate/
                100.txt
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(TRAIN_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(VALIDATE_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(TRAIN_LABEL_FOLDER, exist_ok=True)
    os.makedirs(VALIDATE_LABEL_FOLDER, exist_ok=True)

def save_data(canvas=None, label=None, idx=None, training=True):
    if training:
        image_folder = TRAIN_IMAGE_FOLDER
        label_folder = TRAIN_LABEL_FOLDER
    else:
        image_folder = VALIDATE_IMAGE_FOLDER
        label_folder = VALIDATE_LABEL_FOLDER

    image_path = os.path.join(image_folder, f"{idx:04d}.png")
    label_path = os.path.join(label_folder, f"{idx:04d}.txt")

    cv2.imwrite(image_path, canvas)
    with open(label_path, "w") as f:
        f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}")
    return image_path, label_path

def generate_data(training_num=10, validating_num=10):
    for i in tqdm(range(1, training_num), desc="generating training data..."):
        c, l = synthesize_one()
        save_data(c, l, idx=i, training=True)
    for i in tqdm(range(1, validating_num), desc="generating validate data..."):
        c, l = synthesize_one()
        save_data(c, l, idx=i, training=False)  

def check_data():
    TRAIN_IMG = os.path.join(DATA_FOLDER, 'images', 'train')
    TRAIN_LBL = os.path.join(DATA_FOLDER, 'labels', 'train')

    # 1. image 和 label 數量一致
    print("train images:", len([f for f in os.listdir(TRAIN_IMG) if f.endswith('.png')]))
    print("train labels:", len([f for f in os.listdir(TRAIN_LBL) if f.endswith('.txt')]))

    # 2. 隨便開一個 txt 看格式(一行,五個數字,座標在 0~1)
    sample = sorted(os.listdir(TRAIN_LBL))[0]
    print(sample, "->", open(os.path.join(TRAIN_LBL, sample)).read())

if __name__ == '__main__':
    generate_data(training_num=400, validating_num=100)
    check_data()