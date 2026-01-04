import cv2

class Camera:
    def __init__(self, width=640, height=480, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {source}. Please check if camera is available.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        print(f"Camera initialized successfully: {width}x{height}")

    def read(self):
        success, img = self.cap.read()
        if not success:
            return False, None
        
        # Flip and convert to RGB
        img = cv2.flip(img, 1)
        return True, img

    def release(self):
        self.cap.release()
