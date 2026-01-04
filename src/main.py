import cv2
import yaml
import sys
import os
import numpy as np
import keyboard
import argparse

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.input_layer.camera import Camera
# Lazy import để tránh mediapipe conflicts
# from src.core_engine.face_detector import FaceMeshDetector
# from src.core_engine.corrected_face_detector import CorrectedFaceDetector
from src.core_engine.signal_filters import ExponentialMovingAverage
from src.core_engine.geometry_utils import map_coordinates
from src.logic_layer.mouse_controller import MouseController
from src.logic_layer.gesture_recognizer import GestureRecognizer
from src.ui_layer.hud_overlay import HUDOverlay
from src.ui_layer.system_tray import SystemTrayHandler

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base_path, relative_path)

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Global state
running = True

def toggle_gui():
    """This will be replaced by nonlocal in main()"""  
    pass

def exit_application():
    """Exit the application"""
    global running
    running = False
    print("Exiting application...")

def main():
    global running
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='HeadPose Mouse Control')
        parser.add_argument('--show-gui', action='store_true', help='Start with GUI visible (default is headless)')
        parser.add_argument('--no-tray', action='store_true', help='Disable system tray icon')
        args = parser.parse_args()
        
        # Load Config
        config_path = get_resource_path('configs/default_config.yaml')
        config = load_config(config_path)
        
        # Initial GUI state - default to headless unless --show-gui is specified
        # Check config for default behavior
        start_headless = config.get('system', {}).get('start_headless', True)
        gui_visible = args.show_gui or (not start_headless)
        
        def toggle_gui_callback(visible):
            nonlocal gui_visible
            gui_visible = visible
            print(f"GUI {'Shown' if gui_visible else 'Hidden'}")
        
        def toggle_gui_hotkey():
            nonlocal gui_visible
            gui_visible = not gui_visible
            print(f"GUI {'Shown' if gui_visible else 'Hidden'}")

        # Initialize Modules
        print("Initializing camera...")
        cam = Camera(width=config['system']['cam_width'], height=config['system']['cam_height'])
        print("Initializing face detector...")
        # Lazy import to avoid dependency conflicts
        try:
            from src.core_engine.corrected_face_detector import CorrectedFaceDetector
            detector = CorrectedFaceDetector('models/best_model.pth', model_type='full')
            print("✓ Đang chạy với correction model (độ chính xác cao hơn)")
        except Exception as e:
            print(f"⚠ Không load được correction model: {e}")
            from src.core_engine.face_detector import FaceMeshDetector
            detector = FaceMeshDetector()
            print("→ Chạy với MediaPipe thuần")
        print("Initializing mouse controller...")
        mouse = MouseController()
        gesture_recognizer = GestureRecognizer(config)
        hud = HUDOverlay()
    
        # Smoothing Filters
        smooth_factor = config['system']['smoothing_factor']
        # Alpha = 1 / smoothing_factor roughly for similar effect
        alpha = 1.0 / smooth_factor if smooth_factor > 0 else 1.0
        
        smoother_x = ExponentialMovingAverage(alpha=alpha)
        smoother_y = ExponentialMovingAverage(alpha=alpha)

        frame_reduction = config['system']['frame_reduction']
        nose_id = config['landmarks']['nose_tip']
        
        # Functions for DPI control
        def increase_dpi():
            new_sensitivity = mouse.change_sensitivity(0.2)
            print(f"DPI/Sensitivity increased: {new_sensitivity:.1f}x")
        
        def decrease_dpi():
            new_sensitivity = mouse.change_sensitivity(-0.2)
            print(f"DPI/Sensitivity decreased: {new_sensitivity:.1f}x")
        
        # Setup Hotkeys
        keyboard.add_hotkey('ctrl+shift+h', toggle_gui_hotkey)  # Toggle GUI
        try:
            keyboard.add_hotkey('ctrl+shift+=', increase_dpi)    # Increase DPI/Speed
            keyboard.add_hotkey('ctrl+shift+-', decrease_dpi)   # Decrease DPI/Speed
        except:
            print("⚠ DPI hotkeys không đăng ký được")
        
        # Setup System Tray (if not disabled)
        tray = None
        if not args.no_tray:
            tray = SystemTrayHandler(
                on_toggle_gui=toggle_gui_callback,
                on_exit=exit_application
            )
            tray.run(blocking=False)
            print("System tray icon started. Right-click icon to show/hide video.")

        print("HeadPose Mouse Control Started")
        if not gui_visible:
            print("Running in HEADLESS mode (no video window)")
            print("App is running in system tray. Right-click tray icon to show/hide video.")
        print("Hotkey: Ctrl+Shift+H to toggle video window")
        print("Press Ctrl+C in terminal to exit")

        # Give tray icon time to initialize
        import time
        time.sleep(0.5)


        while running:
            success, img = cam.read()
            if not success:
                break

            img_h, img_w, _ = img.shape
            
            # Process Face
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            landmarks_result = detector.detect_landmarks(rgb_img)
            
            # Create results object compatible with old code
            results = type('obj', (object,), {
                'multi_face_landmarks': [landmarks_result] if landmarks_result else None
            })()

            # Only draw UI if GUI is visible
            if gui_visible:
                hud.draw_active_zone(img, frame_reduction)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    # --- Cursor Control ---
                    nose = landmarks[nose_id]
                    nose_x = int(nose.x * img_w)
                    nose_y = int(nose.y * img_h)

                    if gui_visible:
                        hud.draw_nose(img, nose_x, nose_y)

                    # Map Coordinates
                    x_interp = map_coordinates(nose_x, frame_reduction, img_w - frame_reduction, 0, mouse.screen_width)
                    y_interp = map_coordinates(nose_y, frame_reduction, img_h - frame_reduction, 0, mouse.screen_height)

                    # Smooth
                    curr_x = smoother_x.update(x_interp)
                    curr_y = smoother_y.update(y_interp)

                    # Move Mouse
                    mouse.move(curr_x, curr_y)

                    # --- Scroll Control (nghiêng đầu lên/xuống) ---
                    if config['system'].get('scroll_enabled', False):
                        scroll_amount = gesture_recognizer.detect_head_tilt(nose_y)
                        if scroll_amount != 0:
                            mouse.scroll(scroll_amount)

                    # --- Gesture Control ---
                    action, l_ear, r_ear = gesture_recognizer.detect_blink(landmarks, img_w, img_h)
                    
                    if gui_visible:
                        hud.draw_stats(img, l_ear, r_ear)

                    if action:
                        if action == "left_click":
                            if mouse.click('left', config['system']['click_cooldown']):
                                if gui_visible:
                                    hud.draw_click_feedback(img, action)
                        elif action == "right_click":
                            if mouse.click('right', config['system']['click_cooldown']):
                                if gui_visible:
                                    hud.draw_click_feedback(img, action)

            # Show window only if GUI is visible
            if gui_visible:
                cv2.imshow("Head Pose Mouse Control", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
            else:
                # In headless mode, just wait a bit to reduce CPU usage
                cv2.waitKey(1)

    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if tray is not None:
            tray.stop()

if __name__ == "__main__":
    main()
