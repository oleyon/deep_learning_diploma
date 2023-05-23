import cv2
import os
from pytube import YouTube

# Download YouTube video
yt = YouTube("https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley")  # replace VIDEO_ID with the ID of the YouTube video
yt_stream = yt.streams.filter(file_extension='mp4').first()  # choose a video stream with the mp4 file extension
video_path = yt_stream.download()

# Create output directory for images
output_dir = 'dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open video file and extract frames at a fixed frame rate of 1 frame per second
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Extract frames at a fixed frame rate of 1 frame per second
        if frame_count % int(fps) == 0:
            # Save frame as image file
            filename = os.path.join(output_dir, f'image_{frame_count:05d}.jpg')
            cv2.imwrite(filename, frame)
        frame_count += 1
    else:
        break

# Release video file and print number of frames extracted
cap.release()
print(f'{frame_count} frames extracted and saved to {output_dir}.')