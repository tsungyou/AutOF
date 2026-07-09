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

# 💡 安全防禦機制：當你把滑鼠游標移到螢幕最左上角 (0,0) 時，程式會立刻安全中止
pyautogui.FAILSAFE = True  

LABEL_W, LABEL_H = 1330, 946
CACHE_FILE = "window_regions_cache.json"  # 💡 快取檔案名稱

XYS_ABSOLUTE = {
    "public_card_1": (379, 515, 391, 488),
    "public_card_2": (379, 515, 504, 601),
    "public_card_3": (379, 515, 617, 714),
    "public_card_4": (379, 515, 730, 827),
    "public_card_5": (379, 515, 844, 941),
    "aof_fold_button": ( 835, 930, 870, 1070),
    "aof_allin_button": ( 835, 930, 1090, 1290),
    "hand_area":     (688, 820, 576, 756),
}

SUIT_EMOJI = {
    "club": "♣️", "diamond": "♦️", "heart": "♥️", "spade": "♠️",
}

RANK_STRING = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "10": "T", "11": "J", "12": "Q", "13": "K", "1": "A",
}

class ActionStatus(Enum):
    playable = "playable"
    waiting = "waiting"

class ImageManager:
    def __init__(self):
        print("載入模型(只做一次)...")
        self.model = YOLO(CARD_MODEL_PATH)
        self.window_model = YOLO(WINDOW_MODEL_PATH)
        self.img = None
        self.current_shape = None
        self.px = None
        self.screen_left = 0
        self.screen_top = 0
        self.data = pd.read_csv(STRATEGY_FILE_PATH)
        self.state = None
        
    def update_image(self, img):
        self.img = img
        self.current_shape = img.shape[:2]
        self.px = self._get_px()

    def _get_px(self):
        h, w = self.current_shape
        out = {}
        for name, (y1, y2, x1, x2) in XYS_ABSOLUTE.items():
            ry1, ry2 = min(y1, y2), max(y1, y2)
            rx1, rx2 = min(x1, x2), max(x1, x2)
            out[name] = (int(ry1/LABEL_H*h), int(ry2/LABEL_H*h),
                         int(rx1/LABEL_W*w), int(rx2/LABEL_W*w))
        return out

    def get_roi_from_px(self, name):
        y1, y2, x1, x2 = self.px[name]
        return self.img[y1:y2, x1:x2]

    def _roi_brightness(self, name):
        roi = self.get_roi_from_px(name)
        if roi.size == 0:
            return 0
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()

    def _has_card(self, name, threshold=100):
        return self._roi_brightness(name) > threshold

    def get_board_status(self):
        names = [f"public_card_{i}" for i in range(1, 6)]
        n = sum(self._has_card(c) for c in names)
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n, f"transition({n})")
        return n, street

    def get_action_status(self):
        for b in ['aof_fold_button', "aof_allin_button"]:
            if self._roi_brightness(b) <= 60:
                return ActionStatus.waiting
        return ActionStatus.playable

    def _predict(self, roi):
        if roi.size == 0:
            return None
        r = self.model(roi, verbose=False)[0]
        return (r.names[r.probs.top1], float(r.probs.top1conf))

    def get_public_cards(self):
        out = {}
        for name in [f"public_card_{i}" for i in range(1, 6)]:
            out[name] = self._predict(self.get_roi_from_px(name)) if self._has_card(name) else None
        return out

    def get_hand_cards(self):
        if not self._has_card("hand_area"):
            return {"hand_left": None, "hand_right": None}
        roi = self.get_roi_from_px('hand_area')
        h, w = roi.shape[:2]
        mid = int(w/2 * 0.8)
        left = roi[0:int(h*0.65), 0:mid]
        right = roi[0:int(h*0.65), mid:w]
        return {"hand_left": self._predict(left), "hand_right": self._predict(right)}

    def get_full_state(self):
        n, street = self.get_board_status()
        self.state = {
            "street": street,
            "n_public": n,
            "public_cards": self.get_public_cards(),
            "hand_cards": self.get_hand_cards(),
            "action": self.get_action_status().value,
        }

    def draw_rois_with_result(self):
        vis = self.img.copy()
        for name, (y1, y2, x1, x2) in self.px.items():
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, name, (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for name, res in self.state["public_cards"].items():
            if res:
                y1, y2, x1, x2 = self.px[name]
                cv2.putText(vis, res[0], (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return vis
    
    def set_screen_offset(self, left: int, top: int):
        self.screen_left = left
        self.screen_top = top

    def get_roi_center_px(self, name: str) -> tuple[int, int]:
        if self.px is None:
            raise RuntimeError("請先呼召 update_image()")
        y1, y2, x1, x2 = self.px[name]
        return (x1 + x2) // 2, (y1 + y2) // 2

    def get_roi_screen_pos(self, name: str) -> tuple[int, int]:
        cx, cy = self.get_roi_center_px(name)
        return self.screen_left + cx, self.screen_top + cy

    def click_aof_fold_button(self):
        x, y = self.get_roi_screen_pos("aof_fold_button")
        pyautogui.click(x, y)

    def click_aof_allin_button(self):
        x, y = self.get_roi_screen_pos("aof_allin_button")
        pyautogui.click(x, y)
    
    def run_actions(self, num_of_table: int) -> None:
        if self.state['street'] == "preflop":
            hand_cards_string = holdem_hand_to_string(self.state['hand_cards'])
            d = self.data[self.data['hand'] == hand_cards_string]
            print(f"第[{num_of_table}]桌", datetime.now().strftime("%H:%M:%S"), prettify_cards(self.state['hand_cards']), d.values)
            
            if len(d) > 0:
                sb_freq = d['sb_push_freq'].values[0]
                bb_freq = d['bb_call_freq'].values[0]
                if sb_freq > 0.5 and bb_freq > 0.5:
                    self.click_aof_allin_button()
                elif sb_freq < 0.1 and bb_freq < 0.1:
                    self.click_aof_fold_button()
                else:
                    self.click_aof_fold_button()
            else:
                self.click_aof_fold_button()
    
def prettify_cards(cards):
    prettify_str = ""
    for card in cards.values():
        if card:
            suit, rank = card[0].split('_')
            prettify_str += f"{SUIT_EMOJI[suit]} {rank}"
    return prettify_str.strip()

def holdem_hand_to_string(hand_cards, rank_string_map=None):
    if rank_string_map is None:
        rank_string_map = {
            1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 
            7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"
        }
    try:
        cards = [v[0] for v in hand_cards.values()]
        if len(cards) < 2: return None
        h1_suit, h1_rank_raw = cards[0].split('_')
        h2_suit, h2_rank_raw = cards[1].split('_')
        r1_int, r2_int = int(h1_rank_raw), int(h2_rank_raw)
        if r1_int == 1: r1_int = 14
        if r2_int == 1: r2_int = 14
    except (ValueError, KeyError, IndexError, AttributeError, TypeError):
        return None

    if r1_int >= r2_int:
        hand_cards_string = rank_string_map[r1_int] + rank_string_map[r2_int]
    else:
        hand_cards_string = rank_string_map[r2_int] + rank_string_map[r1_int]

    if r1_int == r2_int: pass
    elif h1_suit == h2_suit: hand_cards_string += "s"
    else: hand_cards_string += "o"
    return hand_cards_string

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
        cv2.imwrite("templates/region_preview.png", shot)
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
    manager = ImageManager()

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
                
                manager.run_actions(num_of_table=index)
                
if __name__ == "__main__":
    main()
