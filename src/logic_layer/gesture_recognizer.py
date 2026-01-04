from src.core_engine.geometry_utils import calculate_ear
import time

class GestureRecognizer:
    def __init__(self, config):
        self.left_eye_ids = config['landmarks']['left_eye']
        self.right_eye_ids = config['landmarks']['right_eye']
        self.ear_threshold = config['thresholds']['ear_blink']
        self.last_nose_y = None
        self.scroll_threshold = 15  # pixels movement to trigger scroll
        
        # Blink detection improvement
        self.blink_frames_required = config['thresholds'].get('blink_frames_required', 2)
        self.click_cooldown = config['system'].get('click_cooldown', 0.5)
        self.left_blink_counter = 0
        self.right_blink_counter = 0
        self.last_click_time = 0
        self.both_eyes_open_required = True  # Yêu cầu cả 2 mắt mở trước khi detect click mới

    def detect_blink(self, landmarks, img_w, img_h):
        left_ear = calculate_ear(landmarks, self.left_eye_ids, img_w, img_h)
        right_ear = calculate_ear(landmarks, self.right_eye_ids, img_w, img_h)

        # Logic: 
        # Left Eye Wink (Right eye open, Left eye closed) -> Left Click
        # Right Eye Wink (Left eye open, Right eye closed) -> Right Click
        
        current_time = time.time()
        action = None
        
        # Kiểm tra cooldown - không cho phép click quá nhanh
        if current_time - self.last_click_time < self.click_cooldown:
            # Reset counters nếu đang trong cooldown
            self.left_blink_counter = 0
            self.right_blink_counter = 0
            return None, left_ear, right_ear
        
        # Kiểm tra cả 2 mắt có mở không (reset state)
        if left_ear > self.ear_threshold and right_ear > self.ear_threshold:
            self.both_eyes_open_required = False
            self.left_blink_counter = 0
            self.right_blink_counter = 0
        
        # Chỉ detect click khi đã qua trạng thái cả 2 mắt mở
        if self.both_eyes_open_required:
            return None, left_ear, right_ear
        
        # Đếm frame nháy mắt trái (Left Click)
        if left_ear < self.ear_threshold and right_ear > self.ear_threshold:
            self.left_blink_counter += 1
            self.right_blink_counter = 0  # Reset counter bên kia
            
            if self.left_blink_counter >= self.blink_frames_required:
                action = "left_click"
                self.left_blink_counter = 0
                self.last_click_time = current_time
                self.both_eyes_open_required = True
        
        # Đếm frame nháy mắt phải (Right Click)
        elif right_ear < self.ear_threshold and left_ear > self.ear_threshold:
            self.right_blink_counter += 1
            self.left_blink_counter = 0  # Reset counter bên kia
            
            if self.right_blink_counter >= self.blink_frames_required:
                action = "right_click"
                self.right_blink_counter = 0
                self.last_click_time = current_time
                self.both_eyes_open_required = True
        
        # Reset counters nếu không có blink nào rõ ràng
        else:
            self.left_blink_counter = 0
            self.right_blink_counter = 0
            
        return action, left_ear, right_ear
    
    def detect_head_tilt(self, nose_y):
        """
        Detect head tilt for scrolling
        Returns: scroll amount (positive = up, negative = down, 0 = no scroll)
        """
        if self.last_nose_y is None:
            self.last_nose_y = nose_y
            return 0
        
        diff = self.last_nose_y - nose_y
        
        # Only scroll if movement is significant
        if abs(diff) > self.scroll_threshold:
            scroll_amount = diff / 5  # Adjust sensitivity
            self.last_nose_y = nose_y
            return scroll_amount
        
        return 0
