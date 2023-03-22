import cv2
import time
import os
# 비디오 파일 열기
cap = cv2.VideoCapture('20230322_171438.mp4')

# 비디오 파일이 성공적으로 열렸는지 확인
if not cap.isOpened():
    print("Cannot open video file")
    exit()

# 프레임 레이트 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30

# 각 프레임에서 이미지를 추출하고 저장할 폴더 경로 설정
output_folder = 'frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 비디오의 총 프레임 수 가져오기
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 프레임 추출 시작 시간 설정
start_time = time.time()

# 다음 이미지를 추출할 프레임 번호 설정
next_frame = 0

# 각 프레임에서 이미지를 추출하고 저장하는 루프
while True:
    # 현재 시간 가져오기
    current_time = time.time()

    # 프레임 추출 시간 계산
    elapsed_time = current_time - start_time

    # 프레임 추출 간격 계산
    frame_interval = int(fps * elapsed_time)

    # 다음 이미지를 추출할 프레임 번호 계산
    next_frame += frame_interval

    # 마지막 프레임까지 이미지를 추출했으면 루프 종료
    if next_frame >= frame_count:
        break

    # 비디오에서 다음 프레임 가져오기
    cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
    ret, frame = cap.read()

    # 프레임 가져오기가 성공했는지 확인
    if not ret:
        print("Cannot read frame ", next_frame)
        continue

    # 이미지를 저장할 파일 이름 설정
    filename = os.path.join(output_folder, f"frame_{next_frame:04}.png")

    # 이미지를 파일로 저장
    cv2.imwrite(filename, frame)

# 비디오 파일 닫기
cap.release()
