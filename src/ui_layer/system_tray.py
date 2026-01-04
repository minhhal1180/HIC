import pystray
from PIL import Image, ImageDraw
import threading

class SystemTrayHandler:
    def __init__(self, on_toggle_gui, on_exit):
        self.on_toggle_gui = on_toggle_gui
        self.on_exit = on_exit
        self.icon = None
        self.gui_visible = False
        
    def create_icon(self):
        """Create a simple icon for system tray"""
        # Create a simple head icon
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Draw a simple head shape (circle)
        draw.ellipse([8, 8, 56, 56], fill=(100, 150, 255), outline=(0, 0, 0), width=2)
        # Draw eyes
        draw.ellipse([20, 24, 28, 32], fill=(0, 0, 0))
        draw.ellipse([36, 24, 44, 32], fill=(0, 0, 0))
        # Draw smile
        draw.arc([18, 28, 46, 48], start=0, end=180, fill=(0, 0, 0), width=2)
        
        return image
    
    def toggle_gui_visibility(self, icon, item):
        """Toggle GUI visibility"""
        self.gui_visible = not self.gui_visible
        self.on_toggle_gui(self.gui_visible)
        
    def exit_app(self, icon, item):
        """Exit application"""
        icon.stop()
        self.on_exit()
    
    def create_menu(self):
        """Create system tray menu"""
        return pystray.Menu(
            pystray.MenuItem(
                "Show/Hide Video", 
                self.toggle_gui_visibility,
                default=True
            ),
            pystray.MenuItem("Exit", self.exit_app)
        )
    
    def run(self, blocking=False):
        """Run system tray icon"""
        image = self.create_icon()
        self.icon = pystray.Icon(
            "HeadPose Mouse",
            image,
            "HeadPose Mouse Control\nPress Ctrl+Shift+H to toggle video",
            self.create_menu()
        )
        
        if blocking:
            # Run in main thread (blocking)
            self.icon.run()
        else:
            # Run in separate thread
            tray_thread = threading.Thread(target=self.icon.run, daemon=True)
            tray_thread.start()
        
    def stop(self):
        """Stop system tray icon"""
        if self.icon:
            self.icon.stop()
