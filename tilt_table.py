import cv2
import numpy as np
import time
from enum import Enum
from ultralytics import YOLO
import mss
import os
import pandas as pd
import pyautogui
from config import CARD_MODEL_PATH, WINDOW_MODEL_PATH, STRATEGY_FILE_PATH
import webbrowser
from datetime import datetime
import subprocess
import json
import sys  # 💡 導入系統標準輸出流模組
from ImageManager import ImageManager, SUIT_EMOJI
# 💡 安全防禦機制：當你把滑鼠游標移到螢幕最左上角 (0,0) 時，程式會立刻安全中止
pyautogui.FAILSAFE = True  

def detect_big_box(sct, model, screen_index=0, retry_interval=10):
    """偵測大框,沒偵測到就重試直到有"""
    while True:
        shot = cv2.cvtColor(np.array(sct.grab(sct.monitors[screen_index])), cv2.COLOR_BGRA2BGR)
        result = model(shot, verbose=False)[0]
        boxes = [b for b in result.boxes if float(b.conf[0]) > 0.5]
        if boxes:
            # 取信心最高的當大框
            best = max(boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            print(f"偵測到大框: ({x1},{y1})-({x2},{y2})")
            return (x1, y1, x2, y2)
        print(f"沒偵測到,{retry_interval}秒後重試...")
        time.sleep(retry_interval)


def split_into_regions(big_box, rows, cols):
    """把大框切成 rows×cols 個小 region(螢幕絕對座標)"""
    x1, y1, x2, y2 = big_box
    cell_w = (x2 - x1) // cols
    cell_h = (y2 - y1) // rows
    regions = []
    for r in range(rows):
        for c in range(cols):
            rx1 = x1 + c * cell_w
            ry1 = y1 + r * cell_h
            regions.append({
                "left": rx1,
                "top": ry1,
                "width": cell_w,
                "height": cell_h,
            })
    return regions

def cache_path(rows, cols):
    """每種行列一個快取檔"""
    return f"window_regions_caches/region_cache_{rows}x{cols}.json"


def get_regions(sct, manager, rows, cols, screen_index=0):
    path = cache_path(rows, cols)

    # 有快取就問要不要用
    if os.path.exists(path):
        while True:
            choice = input(f"發現 {rows}x{cols} 的快取,是否使用?(y/n): ").strip().lower()
            if choice == 'y':
                with open(path) as f:
                    regions = json.load(f)
                print(f"✅ 已載入 {rows}x{cols} 快取,共 {len(regions)} 桌")
                return regions
            elif choice == 'n':
                print("🔄 重新偵測並覆蓋...")
                break
            else:
                print("請輸入 y 或 n")

    # 沒快取,或選擇重新偵測:偵測大框 + 切割
    big_box = detect_big_box(sct, manager.window_model, screen_index=screen_index)
    regions = split_into_regions(big_box, rows, cols)

    # 存快取(蓋過去)
    with open(path, 'w') as f:
        json.dump(regions, f, indent=4)
    print(f"💾 已儲存 {rows}x{cols} 快取至 {path}")
    return regions

def main(screen_index=0):
    manager = ImageManager()

    with mss.MSS() as sct:
        # 1. 先輸入行列
        rows = int(input("幾行 (rows): "))
        cols = int(input("幾列 (cols): "))

        # 2. 依行列取 region(有快取問要不要用,沒有就偵測)
        regions = get_regions(sct, manager, rows, cols, screen_index=screen_index)
        print(f"使用 {len(regions)} 桌")

        # screen init
        for index, region in enumerate(regions):
            frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
            manager.update_image(frame)
            manager.get_full_state()
            manager.set_screen_offset(region["left"], region["top"])
            
            vis = manager.draw_rois_with_result()
            cv2.imwrite("templates/live_roi_{}.png".format(index), vis)

        last_time = 0
        while True:
            if time.time() - last_time < np.random.randint(5, 10):
                time.sleep(0.2)
                continue
            last_time = time.time()

            for index, region in enumerate(regions):
                frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
                manager.update_image(frame)
                manager.get_full_state()
                manager.set_screen_offset(region["left"], region["top"])
                
                manager.run_actions(num_of_table=index)
                
if __name__ == "__main__":
    main()
