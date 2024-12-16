import cv2
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

class VideoDatasetMaker:
    def __init__(self, video_files, train_size=2000, val_size=2000, test_size=2000):
        self.video_files = video_files
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        os.makedirs('train', exist_ok=True)
        os.makedirs('val', exist_ok=True)
        os.makedirs('test', exist_ok=True)

    def extract_and_save_frames(self, set_name, size):
      total_frames_needed = size
      frames_per_video = total_frames_needed // len(self.video_files)
      frame_groups_per_video = frames_per_video // 5  

      saved_frame_count = 0 
      for video_file in self.video_files:
          cap = cv2.VideoCapture(video_file)
          frame_rate = cap.get(cv2.CAP_PROP_FPS)
          frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


          interval = int(frame_rate * 0.2)

          for _ in range(frame_groups_per_video):
              start_frame = random.randint(int(frame_rate) * 180, int(frame_count - interval * 5))
              for i in range(5):
                  frame_idx = start_frame + i * interval
                  if frame_idx >= frame_count:
                      break 
                  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                  ret, frame = cap.read()
                  if not ret:
                      break
                  frame_path = f'{set_name}/{set_name}_{saved_frame_count:05d}.png'
                  cv2.imwrite(frame_path, frame)
                  saved_frame_count += 1

          cap.release()


    def run(self):
        self.extract_and_save_frames('train', self.train_size)
        self.extract_and_save_frames('val', self.val_size)
        self.extract_and_save_frames('test', self.test_size)
