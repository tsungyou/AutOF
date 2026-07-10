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


def select_region_by_clicks():
    """移動滑鼠到兩個角,各按 Enter 記錄座標"""
    input("把滑鼠移到【左上角】,然後按 Enter...")
    x1, y1 = pyautogui.position()
    print(f"左上角: ({x1}, {y1})")

    input("把滑鼠移到【右下角】,然後按 Enter...")
    x2, y2 = pyautogui.position()
    print(f"右下角: ({x2}, {y2})")

    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)

    # 防呆:框太小代表可能沒移動或按錯
    if width < 50 or height < 50:
        print("⚠️ 選取範圍太小,請重新選取")
        return select_region_by_clicks()   # 重來一次

    region = {"left": left, "top": top, "width": width, "height": height}
    print("選取區域:", region)

    # 截一張確認,存檔讓你看框對不對
    with mss.MSS() as sct:
        shot = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
        i = ImageManager(mode='TOURNEY')
        i.update_image(shot)
        i.get_full_state()
        vis = i.draw_rois_with_result()
        cv2.imwrite("templates/region_preview.png", vis)
        i = None
    print("已存 region_preview.png,確認框住的是整個多桌畫面")

    return region

def cache_path(rows, cols):
    """每種行列一個快取檔"""
    return f"window_regions_caches/region_cache_{rows}x{cols}.json"

def get_regions(rows, cols):
    path = cache_path(rows, cols)   # region_cache_2x2.json 之類

    # 有快取問要不要用
    if os.path.exists(path):
        while True:
            choice = input(f"發現 {rows}x{cols} 快取,使用?(y/n): ").strip().lower()
            if choice == 'y':
                with open(path) as f:
                    regions = json.load(f)
                print(f"✅ 載入 {rows}x{cols} 快取,{len(regions)} 桌")
                return regions
            elif choice == 'n':
                break
            else:
                print("請輸入 y 或 n")

    # 沒快取或選重來:手動框 + 切割
    big_region = select_region_by_clicks()      # 滑鼠框大畫面
    big_box = (big_region["left"], big_region["top"],
               big_region["left"] + big_region["width"],
               big_region["top"] + big_region["height"])
    regions = split_into_regions(big_box, rows, cols)

    with open(path, 'w') as f:
        json.dump(regions, f, indent=4)
    print(f"💾 已存 {rows}x{cols} 快取")
    return regions


def main():
    manager = ImageManager(mode='TOURNEY')

    with mss.MSS() as sct:
        # 1. 先輸入行列
        rows = int(input("幾行 (rows): "))
        cols = int(input("幾列 (cols): "))

        # 2. 依行列取 region(有快取問要不要用,沒有就偵測)
        regions = get_regions(rows, cols)
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
                
                manager.run_normal_cash_game_actions(num_of_table=index)
                vis = manager.draw_rois_with_result()
                cv2.imwrite("templates/live_roi_{}.png".format(index), vis)
if __name__ == "__main__":
    main()
