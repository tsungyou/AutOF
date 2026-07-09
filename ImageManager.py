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


class ActionStatus(Enum):
    playable = "playable"
    waiting = "waiting"
    
SUIT_EMOJI = {
    "club": "♣️", "diamond": "♦️", "heart": "♥️", "spade": "♠️",
}


LABEL_W, LABEL_H = 1330, 946
CACHE_FILE = "window_regions_cache.json"  # 💡 快取檔案名稱

REF_H, REF_W = 357, 506   # 參考圖 shape = (H, W)

blinds_area = (
    220 * LABEL_H / REF_H,   # y1 ≈ 583
    250 * LABEL_H / REF_H,   # y2 ≈ 662
    241 * LABEL_W / REF_W,   # x1 ≈ 633
    265 * LABEL_W / REF_W,   # x2 ≈ 697
)
# x1, y1, x2, y2
XYS_ABSOLUTE = {
    "public_card_1": (379, 515, 391, 488),
    "public_card_2": (379, 515, 504, 601),
    "public_card_3": (379, 515, 617, 714),
    "public_card_4": (379, 515, 730, 827),
    "public_card_5": (379, 515, 844, 941),
    "aof_fold_button": ( 835, 930, 870, 1070),
    "aof_allin_button": ( 835, 930, 1090, 1290),
    "hand_area":     (688, 820, 576, 756),
    "blinds_area": blinds_area
}

RANK_STRING = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "10": "T", "11": "J", "12": "Q", "13": "K", "1": "A",
}
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
        self.position_templates = self.get_position_templates()
        
    def get_position_templates(self):
        # 💡 1. 預先準備好你那三張經典截圖的 ROI（請用你原本的截圖路徑）
        # 這裡為了維持辨識穩定，我們轉成灰階（因為字是白的、底是綠的，灰階就能完美區分）
        roi_none = cv2.cvtColor(cv2.imread("models/button.png")[220:250, 241:265], cv2.COLOR_BGR2GRAY)
        roi_05bb = cv2.cvtColor(cv2.imread("models/sb_blinds.png")[220:250, 241:265], cv2.COLOR_BGR2GRAY)
        roi_1bb  = cv2.cvtColor(cv2.imread("models/bb_blinds.png")[220:250, 241:265], cv2.COLOR_BGR2GRAY)

        templates = {
            "Button or Else": roi_none.flatten().astype(np.float32),
            "Small Blind": roi_05bb.flatten().astype(np.float32),
            "Big Blind": roi_1bb.flatten().astype(np.float32)
        }
        return templates
    
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
    
    def get_position_by_blinds_area(self):
        roi = self.get_roi_from_px('blinds_area')
        if roi.size == 0: 
            return "Unknown"
        
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. 高門檻二值化，只抓純白字體
        _, thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
        
        # 2. 尋找所有白色像素點的座標
        white_pixels = np.where(thresh == 255)
        white_pixel_count = len(white_pixels[0])
        
        # 3. 核心邏輯判定
        if white_pixel_count < 10:
            # 白色像素太少，代表根本沒字
            best_match = "Button or Else"
            text_width = 0
        else:
            # 💡 算出白色字體在水平方向 (x軸) 的最左端與最右端
            white_xs = white_pixels[1]
            min_x = np.min(white_xs)
            max_x = np.max(white_xs)
            
            # 這是字體實際佔用的總寬度（像素值）
            text_width = max_x - min_x
            
            # 💡 用「字體寬度佔整個 ROI 寬度的比例」來判定，這樣完全無視視窗縮放！
            roi_width = roi_gray.shape[1]
            width_ratio = (text_width / roi_width) * 100
            
            # 註：0.5 BB 因為多了 "0."，字體會幾乎撐滿整個黑框的左右兩側 (通常 > 75%)
            # 1 BB 因為只有 "1 BB"，且 1 很瘦，左右留白多 (通常 < 65%)
            if width_ratio > 70.0:
                best_match = "Small Blind"
            else:
                best_match = "Big Blind"

        return best_match
    
    def get_full_state(self):
        n, street = self.get_board_status()
        self.state = {
            "street": street,
            "n_public": n,
            "public_cards": self.get_public_cards(),
            "hand_cards": self.get_hand_cards(),
            "action": self.get_action_status().value,
            "position": self.get_position_by_blinds_area(),
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
        # preflop -> big blind -> hu_strategy.csv
        if self.state['street'] == "preflop":
            hand_cards_string = holdem_hand_to_string(self.state['hand_cards'])
            d = self.data[self.data['hand'] == hand_cards_string]
            print(f"第[{num_of_table}]桌", self.state['position'], datetime.now().strftime("%H:%M:%S"), prettify_cards(self.state['hand_cards']), d.values)
            
            if len(d) > 0:
                if self.state['position'] != "Big Blind":
                    self.click_aof_fold_button()
                    return
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