# camera.py
import cv2
import threading
import time

class VideoCamera:
    def __init__(self, src=1):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("❌ Cannot open camera.")
            self.cap.release()
            self.cap = None
            self.running = False
            return

        self.lock = threading.Lock()
        self.running = True
        self.grabbed, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed and frame is not None:
                with self.lock:
                    self.grabbed, self.frame = grabbed, frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if not self.grabbed or self.frame is None or self.frame.size == 0:
                return None
            return self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()
