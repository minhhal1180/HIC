import pyautogui
import time

class MouseController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        self.last_click_time = 0
        self.sensitivity = 1.0  # DPI multiplier (0.5 = chậm, 1.0 = bình thường, 2.0 = nhanh)

    def move(self, x, y):
        # Apply sensitivity/DPI
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        # Calculate offset from center
        offset_x = (x - center_x) * self.sensitivity
        offset_y = (y - center_y) * self.sensitivity
        
        # Apply offset
        new_x = center_x + offset_x
        new_y = center_y + offset_y
        
        # Clamp to screen bounds
        new_x = max(0, min(self.screen_width - 1, new_x))
        new_y = max(0, min(self.screen_height - 1, new_y))
        
        pyautogui.moveTo(new_x, new_y)

    def click(self, button='left', cooldown=0.5):
        current_time = time.time()
        if current_time - self.last_click_time > cooldown:
            if button == 'left':
                pyautogui.click()
            elif button == 'right':
                pyautogui.rightClick()
            self.last_click_time = current_time
            return True
        return False
    
    def scroll(self, amount):
        """Scroll mouse wheel. Positive = scroll up, Negative = scroll down"""
        pyautogui.scroll(int(amount))
    
    def change_sensitivity(self, delta):
        """Change mouse sensitivity/DPI"""
        self.sensitivity = max(0.3, min(3.0, self.sensitivity + delta))
        return self.sensitivity
