import os
import json
import pyautogui
import mss
import cv2
import numpy as np
import time
from RoiConfig import * # 請確保此檔案存在並包含 LABEL_W, LABEL_H, XYS_ABSOLUTE 等
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import threading
import pandas as pd
class MouseController:
    def __init__(self):
        # 建立一個執行緒鎖，確保滑鼠同一時間只會服務一個請求
        self._lock = threading.Lock()

    def click(self, x: int, y: int, table_id: int, action_name: str):
        with self._lock:
            print(f"[MouseController] 執行 -> 桌 {table_id} 點擊: {action_name} at ({x}, {y})")
            
            # 1. 移動並點擊 (加上極短的 duration 可以避免瞬間瞬移導致遊戲不認帳)
            pyautogui.moveTo(x, y, duration=0.05)
            pyautogui.click()
            time.sleep(0.05)
            pyautogui.click()

            # 3. 統一的冷卻時間，避免連續點擊太快導致 UI 卡死
            time.sleep(0.2)

@dataclass
class TableState:
    # --- 1. 客觀視覺特徵 (由 VisionEngine 負責填寫) ---
    is_my_turn: bool = False
    button_position: Optional[str] = None
    active_seats: List[str] = field(default_factory=list)
    all_in_seats: List[str] = field(default_factory=list)
    raw_hand_cards: Optional[Dict[str, str]] = None  # 例如: {'hand_left': 'spade_11', ...}
    
    # --- 2. 撲克邏輯特徵 (由 ActionAnalyzer 負責解析填寫) ---
    hero_position: str = "Unknown"   # 例如: 'SB', 'BB', 'BTN'
    facing_all_in: bool = False      # 前面是否有人 All-in
    hand_str: Optional[str] = None   # 例如: 'AKs', 'J5o'
    debug_positions: Dict[str, str] = field(default_factory=dict)
    
    def is_playable(self) -> bool:
        """快速判斷這個狀態是否可以進行決策"""
        return self.is_my_turn and self.hero_position != "Unknown" and self.hand_str is not None
import pandas as pd
import numpy as np

class StrategyEngine:
    def __init__(self, csv_path: str):
        # 1. 將 hand 設為 index，查詢速度大幅提升
        self.chart = pd.read_csv(csv_path).set_index('hands')
        print("✅ 策略表已載入，索引已建立。")

    def get_action_button(self, state: TableState) -> Optional[str]:
        if not state.is_playable():
            return None

        # 2. 根據狀態決定查表欄位
        col_name = self._get_column_name(state)
        if not col_name or col_name not in self.chart.columns:
            return "aof_fold_button" # 安全保底

        # 3. 高效查詢
        try:
            # 使用 .loc 查詢 Index，確保 hand_str 存在
            action_prob = self.chart.loc[state.hand_str, col_name]
            
            # 4. 決策邏輯
            # 如果機率大於隨機數，則執行 PUSH，否則 FOLD
            if np.random.rand() < action_prob:
                action = "PUSH"
            else:
                action = "FOLD"
                
            print(f"DEBUG: 手牌 {state.hand_str} | 決策: {action} (機率: {action_prob})")
            
            return "aof_allin_button" if action == "PUSH" else "aof_fold_button"
            
        except KeyError:
            print(f"⚠️ 錯誤: 找不到手牌 {state.hand_str} 的策略數據")
            return "aof_fold_button"

    def _get_column_name(self, state: TableState) -> str:
        # 將條件邏輯抽離，程式更乾淨
        if state.facing_all_in:
            return f"{state.hero_position.lower()}_call"
        else:
            return f"{state.hero_position.lower()}_open"

class ActionAnalyzer:
    CLOCKWISE_SEATS = ["my", "left", "top", "right"]
    
    @staticmethod
    def enrich_state(state: TableState) -> TableState:
        """傳入 VisionEngine 產生的 state，將撲克邏輯數據填補進去"""
        if not state.is_my_turn:
            return state # 沒輪到我們，不用算
            
        active_seats = [s for s in ActionAnalyzer.CLOCKWISE_SEATS if s in state.active_seats]
        n_players = len(active_seats)
        btn_pos = state.button_position
        
        if not btn_pos or btn_pos not in active_seats or n_players < 2:
            return state

        btn_idx = active_seats.index(btn_pos)
        seat_to_pos = {}
        
        if n_players == 4:
            positions = ["BTN", "SB", "BB", "UTG"]
            action_order = ["UTG", "BTN", "SB", "BB"]
        elif n_players == 3:
            positions = ["BTN", "SB", "BB"]
            action_order = ["BTN", "SB", "BB"]
        else:
            positions = ["SB", "BB"]
            action_order = ["SB", "BB"]

        for i in range(n_players):
            seat = active_seats[(btn_idx + i) % n_players]
            seat_to_pos[seat] = positions[i]

        state.hero_position = seat_to_pos.get("my", "Unknown")
        state.debug_positions = seat_to_pos
        
        if state.hero_position != "Unknown":
            hero_action_idx = action_order.index(state.hero_position)
            for all_in_seat in state.all_in_seats:
                all_in_pos = seat_to_pos.get(all_in_seat)
                if all_in_pos and action_order.index(all_in_pos) < hero_action_idx:
                    state.facing_all_in = True
                    break

        state.hand_str = ActionAnalyzer.convert_cards_to_string(state.raw_hand_cards)
        return state

    @staticmethod
    def convert_cards_to_string(raw_cards: dict) -> Optional[str]:
        if not raw_cards or "hand_left" not in raw_cards or "hand_right" not in raw_cards:
            return None
        
        c1 = raw_cards["hand_left"]
        c2 = raw_cards["hand_right"]
        if not c1 or not c2: return None
        
        # 解析花色與點數 (例如 'spade_A' -> suit='spade', rank='A')
        suit1, rank1 = c1.split('_')
        suit2, rank2 = c2.split('_')
        
        # 點數對應表，用來比大小 (A > K > Q > J > T)
        rank_value = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
        
        # 確保大牌在前面
        if int(rank2) > int(rank1):
            rank1, rank2 = rank2, rank1
            suit1, suit2 = suit2, suit1
            
        # 判斷是 suited (同花) 還是 offsuit (不同花)
        suffix = ""
        if rank1 == rank2:
            pass # 口袋對子不需要後綴 (如 'AA', 'KK')
        elif suit1 == suit2:
            suffix = "s"
        else:
            suffix = "o"
            
        return f"{rank_value.get(int(rank1), int(rank1))}{rank_value.get(int(rank2), int(rank2))}{suffix}"
    
class VisionEngine:
    def __init__(self, card_model_path):
        print("載入 YOLO 模型中 (全局只載入一次)...")
        # 這裡未來可以加入 WINDOW_MODEL_PATH
        self.card_model = YOLO(card_model_path)
        
    def extract_state(self, img, px_map) -> TableState:
        if img is None:
            return TableState()

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        button_position = self._check_positions_button(img, px_map)
        is_my_turn = self._check_buttons_active(img, px_map)
        active_seats, all_in_seats = self._analyze_players(img_hsv, px_map)
        
        hand_cards = None
        if is_my_turn:
            hand_cards = self._detect_hand_cards(img, px_map)

        # 直接回傳封裝好的物件
        return TableState(
            is_my_turn=is_my_turn,
            button_position=button_position,
            active_seats=active_seats,
            all_in_seats=all_in_seats,
            raw_hand_cards=hand_cards
        )
    # --- 以下是從你原本程式碼搬過來的具體算法 ---
    
    def _check_positions_button(self, img, px_map):
        """
        檢查哪個位置有黃色 Dealer Button。
        回傳: 有黃色按鈕的位置名稱 (str)，若都沒找到則回傳 None。
        """
        # OpenCV 的 HSV 範圍中，黃色的 Hue 大約落在 20~40 之間
        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([40, 255, 255])
        SEATS = ["my", "left", "top", "right"]
        yellow_pixel_counts = {}
        button_position = None
        for s in SEATS:
            y1, y2, x1, x2 = px_map[f"button_{s}"]
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            # 將 ROI 轉換為 HSV 色彩空間 (只轉換 ROI 比轉換整張大圖更省效能)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 製作黃色遮罩
            mask_yellow = cv2.inRange(roi_hsv, lower_yellow, upper_yellow)
            
            # 計算黃色像素的數量
            yellow_pixel_count = cv2.countNonZero(mask_yellow)
            
            # 設定一個閥值 (Threshold)，避免雜訊誤判
            # 假設按鈕至少佔據 20 個像素 (可依據你的 LABEL_W/H 實際裁切大小微調)
            # if yellow_pixel_count > 20: 
            # yellow_pixel_counts[s] = yellow_pixel_count
            if yellow_pixel_count > 100:
                button_position = s
                break
        return button_position
    
    
    def _check_buttons_active(self, img, px_map):
        """檢查按鈕是否亮起 (大於特定亮度代表輪到 Hero)"""
        # 取 Fold 按鈕區域
        if "aof_allin_button" not in px_map:
            return False
            
        y1, y2, x1, x2 = px_map["aof_allin_button"]
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return False
            
        brightness = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()
        return brightness > 60  # 原本設定的閾值

    def _analyze_players(self, img_hsv, px_map):
        """回傳仍在座位上的玩家，以及已經 All-in 的玩家"""
        active_seats = ["my"]  # 預設 Hero 永遠在
        all_in_seats = []
        
        SEATS = ["my", "left", "top", "right"]
        
        # 檢查座位是否有人 (HSV 判定)
        for s in SEATS[1:]:
            y1, y2, x1, x2 = px_map[f"players_{s}"]
            roi_hsv = img_hsv[y1:y2, x1:x2]
            mask1 = cv2.inRange(roi_hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(roi_hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            if cv2.countNonZero(cv2.bitwise_or(mask1, mask2)) > 50:
                active_seats.append(s)
                
        # 檢查誰 All-in
        for s in active_seats:
            y1, y2, x1, x2 = px_map[f"status_box_{s}"]
            roi_hsv = img_hsv[y1:y2, x1:x2]
            mask1 = cv2.inRange(roi_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(roi_hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
            if cv2.countNonZero(cv2.bitwise_or(mask1, mask2)) > 30:
                all_in_seats.append(s)
                
        return active_seats, all_in_seats

    def _detect_hand_cards(self, img, px_map):
        """呼叫 YOLO 模型辨識手牌"""
        y1, y2, x1, x2 = px_map["hand_area"]
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        h, w = roi.shape[:2]
        mid = int(w / 2 * 0.8)
        left_img = roi[0:int(h*0.65), 0:mid]
        right_img = roi[0:int(h*0.65), mid:w]
        
        # 這裡為了簡單，假設 _predict 存在並回傳牌名 (如 'spade_A')
        return {
            "hand_left": self._predict(left_img),
            "hand_right": self._predict(right_img)
        }
        
    def _predict(self, roi_img):
        if roi_img.size == 0: return None
        r = self.card_model(roi_img, verbose=False)[0]
        return r.names[r.probs.top1]

# 1. 座標轉換引擎：專門處理計算，與業務邏輯分開
class RegionFactory:
    @staticmethod
    def get_px_coords(region, label_h, label_w, xys):
        h, w = region['height'], region['width']
        px_map = {}
        center_map = {}
        for name, (y1, y2, x1, x2) in xys.items():
            # 處理座標範圍
            ry1, ry2 = min(y1, y2), max(y1, y2)
            rx1, rx2 = min(x1, x2), max(x1, x2)
            
            # ROI 區域
            px_map[name] = (int(ry1/LABEL_H*h), int(ry2/LABEL_H*h),
                            int(rx1/LABEL_W*w), int(rx2/LABEL_W*w))
            # 中心點 (x, y) - 注意這裡要加上 region 的偏移
            center_map[name] = (
                region['left'] + int((rx1 + rx2) / 2 / LABEL_W * w),
                region['top'] + int((ry1 + ry2) / 2 / LABEL_H * h)
            )
        return px_map, center_map

# 2. 牌桌控制器：只負責該桌的狀態與動作
class PokerTable:
    # 👇 參數多加一個 mouse_controller 👇
    def __init__(self, table_id, region, sct, vision_engine, mouse_controller):
        self.table_id = table_id
        self.region = region
        self.sct = sct
        self.vision = vision_engine
        self.mouse = mouse_controller  # 儲存滑鼠控制器
        
        self.img = None
        self.state = {} 
        self.px_map, self.center_map = RegionFactory.get_px_coords(
            region, LABEL_H, LABEL_W, {**XYS_ABSOLUTE, **BLINDS, **PLAYERS, **STATUS_BOXES, **BUTTONS}
        )

    def update_frame(self):
        raw = self.sct.grab(self.region)
        self.img = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        self.state = self.vision.extract_state(self.img, self.px_map)
        return self.state

    def request_click(self, name):
        if name in self.center_map:
            x, y = self.center_map[name]
            # 👇 改由統一的滑鼠控制器來執行點擊 👇
            self.mouse.click(x, y, self.table_id, name)
            
    # 👇 新增這個函數 👇
    def save_debug_image(self, folder_name="templates"):
        if self.img is None or self.px_map is None:
            return
            
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        # 複製一份圖片以免污染原本用來辨識的圖
        debug_img = self.img.copy() 
        
        # 遍歷所有的座標並畫框
        for name, (y1, y2, x1, x2) in self.px_map.items():
            # 畫綠色的外框
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在框框左上角寫上 ROI 的名稱
            cv2.putText(debug_img, name, (x1, max(y1-5, 12)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 畫出中心點 (用來確認滑鼠會點在哪裡)
        for name, (cx, cy) in self.center_map.items():
            # 注意：center_map 存的是螢幕絕對座標，要畫在局部截圖上需要扣掉 region 的偏移
            local_x = cx - self.region['left']
            local_y = cy - self.region['top']
            cv2.circle(debug_img, (local_x, local_y), 3, (0, 0, 255), -1) # 紅色小圓點
                        
        filename = os.path.join(folder_name, f"table_{self.table_id}_debug.png")
        cv2.imwrite(filename, debug_img)
        print(f"✅ 已儲存測試圖: {filename}")
        
# 3. 區域擷取器：負責多桌配置
class RegionCapturer:
    def __init__(self):
        self.regions = []
        self._setup()

    def _setup(self):
        inp = input("輸入桌數行列 (例如 2x2): ")
        rows, cols = map(int, inp.split("x"))
        path = f"window_regions_caches/region_cache_{rows}x{cols}.json"
        
        if os.path.exists(path):
            if input("發現快取，使用?(y/n): ") == 'y':
                with open(path, 'r') as f: self.regions = json.load(f)
                return

        print("請依序選取大區域的【左上】與【右下】...")
        input("按 Enter 開始選取左上..."); x1, y1 = pyautogui.position()
        input("按 Enter 開始選取右下..."); x2, y2 = pyautogui.position()
        
        # 自動切分
        cell_w = abs(x2 - x1) // cols
        cell_h = abs(y2 - y1) // rows
        left, top = min(x1, x2), min(y1, y2)
        
        for r in range(rows):
            for c in range(cols):
                self.regions.append({"left": left + c*cell_w, "top": top + r*cell_h, 
                                     "width": cell_w, "height": cell_h})
        with open(path, 'w') as f: json.dump(self.regions, f)

if __name__ == "__main__":
    from RoiConfig import CARD_MODEL_PATH, STRATEGY_CSV_PATH
    
    capturer = RegionCapturer()
    vision_engine = VisionEngine(CARD_MODEL_PATH)
    strategy_engine = StrategyEngine(STRATEGY_CSV_PATH)  
    mouse_controller = MouseController() # <--- 實例化全域唯一的滑鼠控制器
    
    with mss.mss() as sct:
        # 記得把 mouse_controller 傳進去
        tables = [PokerTable(i, reg, sct, vision_engine, mouse_controller) for i, reg in enumerate(capturer.regions)]
        
        print(f"啟動 {len(tables)} 桌監控...")
        
        is_first_run = True 
        
        try:
            while True:
                for table in tables:
                    # 1. 視覺引擎擷取客觀狀態
                    raw_state = table.update_frame() 
                    
                    if is_first_run:
                        table.save_debug_image()
                    
                    # 2. 邏輯引擎補充撲克資訊
                    state = ActionAnalyzer.enrich_state(raw_state)
                    
                    # 3. 決策層
                    button_name = strategy_engine.get_action_button(state)
                    
                    # 4. 執行層
                    if button_name:
                        print(f"桌 {table.table_id} | 位置: {state.hero_position} | 手牌: {state.hand_str} | 面對 All-in: {state.facing_all_in}")
                        table.request_click(button_name)
                        
                    time.sleep(0.5) 
                if is_first_run:
                    is_first_run = False
                    
                time.sleep(7.0)
        except KeyboardInterrupt:
            print("結束執行")