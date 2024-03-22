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

        # 프레임을 저장할 디렉토리 생성
        os.makedirs('train', exist_ok=True)
        os.makedirs('val', exist_ok=True)
        os.makedirs('test', exist_ok=True)

    def extract_and_save_frames(self, set_name, size):
      total_frames_needed = size
      frames_per_video = total_frames_needed // len(self.video_files)
      frame_groups_per_video = frames_per_video // 5  # 한 그룹에 5개의 프레임이 필요

      saved_frame_count = 0  # 저장된 프레임의 수를 추적하기 위한 카운터
      for video_file in self.video_files:
          cap = cv2.VideoCapture(video_file)
          frame_rate = cap.get(cv2.CAP_PROP_FPS)
          frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

          # 0.1초 간격으로 프레임을 저장하기 위한 계산
          interval = int(frame_rate * 0.2)

          for _ in range(frame_groups_per_video):
              # 랜덤 시작점 결정
              start_frame = random.randint(int(frame_rate) * 180, int(frame_count - interval * 5))
              for i in range(5):
                  frame_idx = start_frame + i * interval
                  if frame_idx >= frame_count:
                      break  # 프레임 인덱스가 비디오의 총 프레임 수를 초과하면 중단
                  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                  ret, frame = cap.read()
                  if not ret:
                      break
                  # 프레임 저장, 파일명에 저장 순서 반영
                  frame_path = f'{set_name}/{set_name}_{saved_frame_count:05d}.png'
                  cv2.imwrite(frame_path, frame)
                  saved_frame_count += 1

          cap.release()


    def run(self):
        # 각 세트별로 프레임 추출 및 저장
        self.extract_and_save_frames('train', self.train_size)
        self.extract_and_save_frames('val', self.val_size)
        self.extract_and_save_frames('test', self.test_size)
