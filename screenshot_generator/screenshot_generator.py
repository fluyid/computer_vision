import os
import time
import cv2 as cv
import numpy as np
import mss
import pygetwindow as gw

SAVE_DIR = "dataset/images"
CAPTURE_INTERVAL = 2.0

os.makedirs(SAVE_DIR, exist_ok=True)

class LeagueCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.find_league_window()
        self.frame_count = 0
        self.start_time = time.time()

    def find_league_window(self):
        monitor = self.sct.monitors[1]

        print("Using monitor capture:")
        print(monitor)

        return monitor

    def is_alive(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        avg_sat = np.mean(hsv[:, :, 1])
        return avg_sat > 20

    def is_shop_open(self, frame):
        # shop has a large dark semi-transparent overlay
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < 40

    def is_scoreboard_open(self, frame):
        # Scoreboard has strong horizontal lines and structured patterns
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # edges = cv.Canny(gray, 50, 150)
        # edge_density = np.sum(edges) / edges.size
        # return edge_density > 0.30
        return False
    def is_after_first_minute(self):
        elapsed_time = time.time() - self.start_time
        return elapsed_time > 60

    def capture_frame(self):
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        return frame

    def save_frame(self, frame):
        filename = f"{SAVE_DIR}/frame_{self.frame_count:06d}.png"
        cv.imwrite(filename, frame)
        print(f"Frame saved to {filename}")
        self.frame_count += 1

    def run(self):
        print("Capture started")
        while True:
            frame = self.capture_frame()
            if not self.is_after_first_minute():
                print("Skipping first minute")
                time.sleep(CAPTURE_INTERVAL)
                continue
            if not self.is_alive(frame):
                print("Skipping: dead")
                time.sleep(CAPTURE_INTERVAL)
                continue
            if self.is_shop_open(frame):
                print("Skipping: shop open")
                time.sleep(CAPTURE_INTERVAL)
                continue
            if self.is_scoreboard_open(frame):
                print("Skipping: scoreboard open")
                time.sleep(CAPTURE_INTERVAL)
                continue
            self.save_frame(frame)
            time.sleep(CAPTURE_INTERVAL)

            print("Alive:", self.is_alive(frame))
            print("Shop:", self.is_shop_open(frame))
            print("Scoreboard:", self.is_scoreboard_open(frame))


if __name__ == "__main__":
    capture = LeagueCapture()
    capture.run()
