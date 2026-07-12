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
import os
import webbrowser
from datetime import datetime
import subprocess

"""
Redetect the window regions every time, without saving the cache to file.
"""


LABEL_W, LABEL_H = 1330, 946

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
    "club": "♣️",
    "diamond": "♦️",
    "heart": "♥️",
    "spade": "♠️",
}

RANK_STRING = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "T",
    "11": "J",
    "12": "Q",
    "13": "K",
    "1": "A",
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
        """餵入新的一幀(BGR numpy array)"""
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
        # for b in ['fold_button', "check_button", "bet_button"]:
        for b in ['aof_fold_button', "aof_allin_button"]: # AOF
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

    def draw_rois(self):
        """把所有 ROI 框畫在當前畫面上,回傳供顯示"""
        vis = self.img.copy()
        for name, (y1, y2, x1, x2) in self.px.items():
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, name, (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return vis
    def draw_rois_with_result(self):
        vis = self.img.copy()
        for name, (y1, y2, x1, x2) in self.px.items():
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, name, (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 把辨識出的牌名畫在對應公牌 ROI 上
        for name, res in self.state["public_cards"].items():
            if res:
                y1, y2, x1, x2 = self.px[name]
                cv2.putText(vis, res[0], (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return vis
    
    # pyautogui
    def set_screen_offset(self, left: int, top: int):
        """設定截圖區域在螢幕上的左上角，與 select_region() 的 left/top 對應"""
        self.screen_left = left
        self.screen_top = top

    def get_roi_center_px(self, name: str) -> tuple[int, int]:
        """回傳 ROI 在截圖內的中心點 (x, y)"""
        if self.px is None:
            raise RuntimeError("請先呼叫 update_image()")
        y1, y2, x1, x2 = self.px[name]
        return (x1 + x2) // 2, (y1 + y2) // 2

    def get_roi_screen_pos(self, name: str) -> tuple[int, int]:
        """回傳 ROI 中心在螢幕上的絕對座標"""
        cx, cy = self.get_roi_center_px(name)
        return self.screen_left + cx, self.screen_top + cy

    def click_aof_fold_button(self):
        """點擊 aof_fold_button 中心"""
        x, y = self.get_roi_screen_pos("aof_fold_button")
        pyautogui.click(x, y)

    def click_aof_allin_button(self):
        """點擊 aof_allin_button 中心"""
        x, y = self.get_roi_screen_pos("aof_allin_button")
        pyautogui.click(x, y)
    
    def run_actions(self, num_of_table: int):
        if self.state['street'] == "preflop":
            hand_cards_string = holdem_hand_to_string(self.state['hand_cards'])
            d = self.data[self.data['hand'] == hand_cards_string]
            print(f"第[{num_of_table}]桌", datetime.now().strftime("%H:%M:%S"), prettify_cards(self.state['hand_cards']), d.values)
            if len(d) > 0:
                # fold losing hands
                if d['sb_push_freq'].values[0] > 0.5 and d['bb_call_freq'].values[0] > 0.5:
                    self.click_aof_allin_button()
                elif d['sb_push_freq'].values[0] < 0.1 and d['bb_call_freq'].values[0] < 0.1:
                    self.click_aof_fold_button()
                else:
                    self.click_aof_fold_button()


def select_region(index=1):
    """直接用座標指定截圖區域,靠即時預覽調整,不用 selectROI"""
    # 先給一個預設區域,使用者看預覽再調這四個數字
    with mss.MSS() as sct:
        monitor = sct.monitors[index]      # 截螢幕2(影片所在的那個)
        shot = np.array(sct.grab(monitor))
        region = {
            "left": monitor["left"],  # 螢幕2起點 + 牌桌在螢幕內的 x 偏移
            "top": monitor["top"]+25,    # 螢幕2起點 + 牌桌在螢幕內的 y 偏移
            "width": int(1330 * 1080 / 946),
            "height": int(946 * 1080 / 946) - 25,
        }
    return region

def prettify_cards(cards):
    prettify_str = ""
    for card in cards.values():
        if card:
            suit, rank = card[0].split('_')
            prettify_str += f"{SUIT_EMOJI[suit]} {rank} "
    return prettify_str

def holdem_hand_to_string(hand_cards, rank_string_map=None):
    """
    德州撲克手牌轉換工具 (Texas Hold'em Hand to 169-Matrix String)

    【功能簡介】
    本函式將具體花色的兩張手牌（例如：黑桃A、紅心K），轉化為 169 核心範圍矩陣對應的靜態字串。
    轉換邏輯嚴格遵循撲克組合數學：
    1. 口袋對子 (Pairs) -> 輸出兩碼字串，如 "AA", "KK", "22"。
    2. 同花手牌 (Suited) -> 點數大者在前，結尾加上 "s"，如 "AKs", "JTs"。
    3. 不同花手牌 (Offsuited) -> 點數大者在前，結尾加上 "o"，如 "AQo", "72o"。

    【點數權重修正 (Ace-High)】
    如果傳入的牌組中，Ace 的點數是以 '1' 表示，本函式會自動將其權重提升至 14（大於 King 的 13），
    以確保大點數永遠排在字串前方，避免生成像是 "2As" 這種非標準字串。

    【輸入格式範例】
    hand_cards = {
        "card1": ["SPADE_1"],   # 代表黑桃 A (點數 1)
        "card2": ["HEART_13"]   # 代表紅心 K (點數 13)
    }
    👉 預期輸出: "AKs" 或 "AKo" 視花色而定。
    """
    # 預設的點數對應表，將數字轉為標準撲克字元
    if rank_string_map is None:
        rank_string_map = {
            1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 
            7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"
        }
    
    try:
        # 安全地從字典中取出兩張牌的字串 (例如 "SPADE_1")
        cards = [v[0] for v in hand_cards.values()]
        if len(cards) < 2:
            return None
        
        # 解析花色與點數
        h1_suit, h1_rank_raw = cards[0].split('_')
        h2_suit, h2_rank_raw = cards[1].split('_')
        
        r1_int = int(h1_rank_raw)
        r2_int = int(h2_rank_raw)
        
        # 核心優化：如果 Ace 是 1，直接將其權重升格為 14 (Ace-High)
        if r1_int == 1: r1_int = 14
        if r2_int == 1: r2_int = 14
        
    except (ValueError, KeyError, IndexError, AttributeError, TypeError):
        return None

    # 1. 決定點數部分的字串組合 (大點數永遠在前)
    if r1_int >= r2_int:
        hand_cards_string = rank_string_map[r1_int] + rank_string_map[r2_int]
    else:
        hand_cards_string = rank_string_map[r2_int] + rank_string_map[r1_int]

    # 2. 決定後綴類型 (對子不加後綴、同花加 s、不同花加 o)
    if r1_int == r2_int:
        pass  # 口袋對子，不需要後綴 (例如 "AA")
    elif h1_suit == h2_suit:
        hand_cards_string += "s"  # 同花 (Suited)
    else:
        hand_cards_string += "o"  # 不同花 (Offsuited)

    return hand_cards_string

def wait_for_start():
    """等使用者準備好、輸入 y 才開始"""
    while True:
        user = input("牌桌都開好了嗎?輸入 y 開始(q 離開): ").strip().lower()
        if user == "y":
            print("開始執行...")
            return True
        elif user == "q":
            print("已取消")
            return False
        else:
            print("請輸入 y 或 q")

def main():
    manager = ImageManager()
    
    if not wait_for_start():
        return
    
    with mss.MSS() as sct:
        # init screen windows
        monitor = sct.monitors[0]
        shot = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)  # 4通道→3通道
        result = manager.window_model(shot, verbose=False)[0]
        window_regions = []
        

        
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if conf > 0.5:
                window_regions.append({
                    "left": int(x1),
                    "top": int(y1),
                    "width": int(x2-x1),
                    "height": int(y2-y1),
                })
            print("牌桌數: ", len(window_regions))
            
            
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cv2.rectangle(shot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite("templates/screen_shot.png", shot)
        
        # strategy looper
        last_time = 0
        pic_time = 0
        while True:
            if time.time() - last_time < np.random.randint(5, 10):
                time.sleep(0.2)
                continue
            last_time = time.time()

            for index, region in enumerate(window_regions):
                frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
                manager.update_image(frame)
                manager.get_full_state()
                manager.set_screen_offset(region["left"], region["top"])
                manager.run_actions(num_of_table=index)
                vis = manager.draw_rois_with_result()
                if time.time() - pic_time > 60:
                    pic_time = time.time()
                    cv2.imwrite("templates/live_roi_{}.png".format(index), vis)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

