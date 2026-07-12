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

REF_BETTING_OPTIONS_W, REF_BETTING_OPTIONS_H = 675, 488
betting_1 = (
    405 * LABEL_H / REF_BETTING_OPTIONS_H,
    425 * LABEL_H / REF_BETTING_OPTIONS_H,
    410 * LABEL_W / REF_BETTING_OPTIONS_W,
    440 * LABEL_W / REF_BETTING_OPTIONS_W,
)

BLINDS = {
    "blinds_right": (162, 177, 404, 418),
    "blinds_my": (220, 235, 248, 262),
    "blinds_left": (162, 177, 87, 101),
    "blinds_top": (93, 108, 248, 262),
}


PLAYERS = {
    "players_left": (135, 170, 10, 70),
    "players_top": (35, 70, 225, 285),
    "players_right": (135, 170, 435, 495)
}

STATUS_BOXES = {
    "status_box_my": (325, 340, 240, 270),
    "status_box_left": (180, 195, 25, 55),
    "status_box_top": (80, 95, 240, 270),
    "status_box_right": (180, 195, 450, 480)
}

BLINDS = {
    i: (int(j * LABEL_H / REF_H), int(k * LABEL_H / REF_H), int(l * LABEL_W / REF_W), int(m * LABEL_W / REF_W)) for i, (j, k, l, m) in BLINDS.items()
}

PLAYERS = {
    i: (int(j * LABEL_H / REF_H), int(k * LABEL_H / REF_H), int(l * LABEL_W / REF_W), int(m * LABEL_W / REF_W)) for i, (j, k, l, m) in PLAYERS.items()
}

STATUS_BOXES = {
    i: (int(j * LABEL_H / REF_H), int(k * LABEL_H / REF_H), int(l * LABEL_W / REF_W), int(m * LABEL_W / REF_W)) for i, (j, k, l, m) in STATUS_BOXES.items()
}





# --- 翻牌前 (Pre-flop) 行動順序映射 ---
# 數字越小代表越先行動，用來判斷 All-in 是在 Hero 之前還是之後
PREFLOP_ORDER = {
    "UTG": 1,
    "button": 2,
    "small blind": 3,
    "big blind": 4,
    "Unknown": 99
}

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
}

XYS_TORUNEY = {
    "public_card_1": (379, 515, 391, 488),
    "public_card_2": (379, 515, 504, 601),
    "public_card_3": (379, 515, 617, 714),
    "public_card_4": (379, 515, 730, 827),
    "public_card_5": (379, 515, 844, 941),
    "fold_button":   (835, 930, 811, 973),
    "check_button":  (835, 930, 982, 1144),
    "bet_button":    (835, 930, 1152, 1315),
    "balance_label": (863, 886, 600, 730),
    "hand_area":     (688, 820, 576, 756),
    "betting_1": betting_1
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


# --- 顏色偵測輔助函數 ---
def check_has_chip(hsv_img, roi_coords, threshold=50):
    y1, y2, x1, x2 = roi_coords
    mask = cv2.inRange(hsv_img[y1:y2, x1:x2], np.array([35, 50, 50]), np.array([85, 255, 255]))
    return cv2.countNonZero(mask) > threshold

def check_is_occupied(hsv_img, roi_coords, threshold=50):
    y1, y2, x1, x2 = roi_coords
    roi_hsv = hsv_img[y1:y2, x1:x2]
    mask1 = cv2.inRange(roi_hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(roi_hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    return cv2.countNonZero(cv2.bitwise_or(mask1, mask2)) > threshold

def check_player_action(hsv_img, roi_coords, threshold=30):
    y1, y2, x1, x2 = roi_coords
    roi_hsv = hsv_img[y1:y2, x1:x2]
    mask1 = cv2.inRange(roi_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(roi_hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    return cv2.countNonZero(cv2.bitwise_or(mask1, mask2)) > threshold

def get_position_name(offset, n_players):
    """根據與 SB 的相對距離回傳位置名稱"""
    if offset == 0: return "small blind"
    elif offset == 1: return "big blind"
    elif offset == (-1 % n_players): return "button"
    elif offset == (-2 % n_players) and n_players == 4: return "UTG"
    return "Unknown"

class ImageManager:
    def __init__(self, mode='AOF'):
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
        self.xys = None
        self.mode = mode
        self.init_xys()
        
    def init_xys(self):
        self.xys = {}
        self.xys.update(XYS_ABSOLUTE if self.mode == 'AOF' else XYS_TORUNEY)
        self.xys.update(BLINDS)
        self.xys.update(PLAYERS)
        self.xys.update(STATUS_BOXES)
    
    def update_image(self, img):
        self.img = img
        self.current_shape = img.shape[:2]
        self.px = self._get_px()

    def _get_px(self):
        h, w = self.current_shape
        out = {}
        for name, (y1, y2, x1, x2) in self.xys.items():
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
    
    def analyze_poker_state(self):
        img = self.img
        if img is None: return {"error": "無法讀取影像"}
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. 定義標準化的座位順序 (my 永遠是 Hero)
        # 用一個列表維護順序，確保後續索引一致
        SEATS = ["my", "left", "top", "right"]
        
        # 抓出有效玩家 (Hero 預設存在)
        # active_seat_keys 只存 ["my", "left", ...] 這些 key
        active_seat_keys = ["my"]
        for s in SEATS[1:]:
            if check_is_occupied(img_hsv, self.px[f"players_{s}"]):
                active_seat_keys.append(s)
        
        n_players = len(active_seat_keys)

        # 2. 依照剛剛定義的 active_seat_keys 順序，檢查盲注位置
        chips_status = []
        for s in active_seat_keys:
            chips_status.append(check_has_chip(img_hsv, self.px[f"blinds_{s}"]))
            
        total_chips_on_table = sum(chips_status)

        # 3. 尋找 SB 並計算位置
        sb_idx = -1
        if n_players >= 2:
            for i in range(n_players):
                # 假設連續兩家有籌碼即為 SB 和 BB
                if chips_status[i] and chips_status[(i + 1) % n_players]:
                    sb_idx = i
                    break

        # 建立座位對應表
        seat_to_poker_pos = {}
        hero_position = "Unknown"
        
        if sb_idx != -1:
            for idx, seat_key in enumerate(active_seat_keys):
                offset = (idx - sb_idx) % n_players
                pos_name = get_position_name(offset, n_players)
                seat_to_poker_pos[seat_key] = pos_name
                if seat_key == "my":
                    hero_position = pos_name

        # 4. 判斷 All-in 狀態 (只遍歷 active_seat_keys)
        all_in_players = []
        for seat_key in active_seat_keys:
            if check_player_action(img_hsv, self.px[f"status_box_{seat_key}"]):
                all_in_players.append({
                    "seat": seat_key,
                    "poker_pos": seat_to_poker_pos.get(seat_key, "Unknown")
                })

        # 5. 決策邏輯 (Hero 判斷使用 "my")
        action_advice = "進行中"
        
        # 判斷 Hero 是否 All-in (檢查 seat_key 為 "my" 的成員是否在 all_in_list)
        is_hero_allin = any(p["seat"] == "my" for p in all_in_players)
        opponents_all_in = [p for p in all_in_players if p["seat"] != "my"]

        if is_hero_allin:
            action_advice = "Hero 已經 All-in"
        elif opponents_all_in:
            hero_order = PREFLOP_ORDER.get(hero_position, 99)
            advice_details = [f"{o['poker_pos']} All-in" for o in opponents_all_in]
            action_advice = f"遇到對手 All-in: " + "，".join(advice_details)
        
        return {
            "n_players": n_players,
            "total_chips": total_chips_on_table,
            "hero_position": hero_position,
            "all_in_count": len(all_in_players),
            "action_advice": action_advice
        }

    def click_aof_fold_button(self):
        x, y = self.get_roi_screen_pos("aof_fold_button")
        pyautogui.click(x, y)

    def click_aof_allin_button(self):
        x, y = self.get_roi_screen_pos("aof_allin_button")
        pyautogui.click(x, y)
    
    
    ##### Tourney, Normal Cash Game
    def click_normal_mode_betting_1_button(self):
        x, y = self.get_roi_screen_pos("betting_1")
        pyautogui.click(x, y)
    
    def click_normal_mode_fold_button(self):
        x, y = self.get_roi_screen_pos("fold_button")
        pyautogui.click(x, y)
    
    def click_normal_mode_check_button(self):
        x, y = self.get_roi_screen_pos("check_button")
        pyautogui.click(x, y)
    
    def click_normal_mode_bet_button(self):
        x, y = self.get_roi_screen_pos("bet_button")
        pyautogui.click(x, y)
    
    def run_normal_cash_game_actions(self, num_of_table: int) -> None:
        hand_cards_string = holdem_hand_to_string(self.state['hand_cards'])
        if hand_cards_string is not None:
            if len(hand_cards_string) != 2:
                cardA, cardB, _ = hand_cards_string
            else:
                cardA, cardB = hand_cards_string

            if cardA in ['A', 'K', 'Q', 'J', 'T'] and cardB in ['A', 'K', 'Q', 'J', 'T']:
                self.click_normal_mode_betting_1_button()
                self.click_normal_mode_bet_button()
                time.sleep(0.5)
                self.click_normal_mode_check_button()
                time.sleep(0.5)
                self.click_normal_mode_check_button()
                print(datetime.now().strftime("%H:%M:%S"), cardA, cardB, "bet")
            else:
                self.click_normal_mode_fold_button()
                print(datetime.now().strftime("%H:%M:%S"), cardA, cardB, "fold")
                
    def run_actions(self, num_of_table: int) -> None:
        # preflop -> big blind -> hu_strategy.csv
        if self.state['street'] == "preflop":
            hand_cards_string = holdem_hand_to_string(self.state['hand_cards'])
            d = self.data[self.data['hand'] == hand_cards_string]
            print(f"第[{num_of_table}]桌", self.state['position'], datetime.now().strftime("%H:%M:%S"), prettify_cards(self.state['hand_cards']), d.values)
            
            if len(d) > 0:
                if self.state['position'] != "Button or Else":
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
                
    def run_tilted_actions(self, num_of_table: int) -> None:
        state = self.analyze_poker_state()
        c = check_player_action(self.img, self.px["aof_allin_button"])
        print(c)
        print(state)