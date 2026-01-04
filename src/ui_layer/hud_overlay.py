import cv2

class HUDOverlay:
    def __init__(self):
        pass

    def draw_active_zone(self, img, frame_reduction):
        h, w, _ = img.shape
        cv2.rectangle(img, (frame_reduction, frame_reduction), 
                      (w - frame_reduction, h - frame_reduction), 
                      (0, 255, 0), 2)

    def draw_nose(self, img, x, y):
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)

    def draw_stats(self, img, left_ear, right_ear):
        cv2.putText(img, f'L_EAR: {left_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(img, f'R_EAR: {right_ear:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def draw_click_feedback(self, img, action):
        if action == "left_click":
            cv2.putText(img, "LEFT CLICK", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif action == "right_click":
            cv2.putText(img, "RIGHT CLICK", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
