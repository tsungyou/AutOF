from enum import Enum

CARD_MODEL_PATH = "models/card_model.pt"
STRATEGY_CSV_PATH = "strategy/my_custom_aof_strategy.csv"
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

BUTTONS = {
    "button_my": (245, 257, 215, 229),
    "button_left": (192, 204, 78, 92),
    "button_top": (95, 107, 224, 238),
    "button_right": (193, 205, 415, 429)
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

BUTTONS = {
    i: (int(j * LABEL_H / REF_H), int(k * LABEL_H / REF_H), int(l * LABEL_W / REF_W), int(m * LABEL_W / REF_W)) for i, (j, k, l, m) in BUTTONS.items()
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