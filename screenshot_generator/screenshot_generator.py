import os
import time
import cv2 as cv
import numpy as np
import mss
import pygetwindow as gw

BASE_SAVE_DIR = "dataset/images"
CAPTURE_INTERVAL = input(float("Enter how many seconds to capture before saving: "))

os.makedirs(BASE_SAVE_DIR, exist_ok=True)


class LeagueCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.find_league_window()

        self.game_dir = self.create_new_game_folder()

        # Continue global frame numbering
        self.frame_count = self.get_global_frame_count()

        self.start_time = time.time()

        print(f"Saving to folder: {self.game_dir}")
        print(f"Starting frame number: {self.frame_count}")

    def get_global_frame_count(self):
        max_frame = -1

        for game_folder in os.listdir(BASE_SAVE_DIR):
            folder_path = os.path.join(BASE_SAVE_DIR, game_folder)

            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                if file.startswith("frame_") and file.endswith(".png"):
                    try:
                        number = int(file.replace("frame_", "").replace(".png", ""))
                        if number > max_frame:
                            max_frame = number
                    except:
                        pass

        return max_frame + 1

    def create_new_game_folder(self):
        existing = os.listdir(BASE_SAVE_DIR)

        game_numbers = []

        for folder in existing:
            if folder.startswith("game_"):
                try:
                    number = int(folder.split("_")[1])
                    game_numbers.append(number)
                except:
                    pass

        if len(game_numbers) == 0:
            next_game_number = 1
        else:
            next_game_number = max(game_numbers) + 1

        new_folder = os.path.join(BASE_SAVE_DIR, f"game_{next_game_number}")
        os.makedirs(new_folder, exist_ok=True)

        return new_folder

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
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < 40

    def is_scoreboard_open(self, frame):
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
        filename = os.path.join(
            self.game_dir,
            f"frame_{self.frame_count:06d}.png"
        )

        cv.imwrite(filename, frame)

        print(f"Saved: {filename}")

        self.frame_count += 1

    def run(self):
        print("Capture started")

        while True:
            frame = self.capture_frame()

            if not self.is_after_first_minute():
                time.sleep(CAPTURE_INTERVAL)
                continue

            if not self.is_alive(frame):
                time.sleep(CAPTURE_INTERVAL)
                continue

            if self.is_shop_open(frame):
                time.sleep(CAPTURE_INTERVAL)
                continue

            if self.is_scoreboard_open(frame):
                time.sleep(CAPTURE_INTERVAL)
                continue

            self.save_frame(frame)

            time.sleep(CAPTURE_INTERVAL)


if __name__ == "__main__":
    capture = LeagueCapture()
    capture.run()
