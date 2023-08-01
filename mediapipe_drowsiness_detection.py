import cv2
import datetime
import numpy as np
from time import time
import mediapipe as mp
from os import mkdir, listdir, remove
from os.path import isdir, join, isfile
import mysql.connector
import csv
from pygame import mixer

mixer.init()
sound=mixer.Sound("buzzer3.mp3")

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database = "radar"
)

mycursor = mydb.cursor()

def calculate_EAR(eye):
	A = np.linalg.norm(eye[2] - eye[3])
	B = np.linalg.norm(eye[4] - eye[5])
	C = np.linalg.norm(eye[6] - eye[7])
	D = np.linalg.norm(eye[0] - eye[1])
	ear = (A + B + C) / (3.0 * D)
	return ear


def get_eyes_points(image, face_mesh):
    global left_eye_points, right_eye_points, LEFT_EYE, RIGHT_EYE, Threshold
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
        return image, 'nao'

    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    annotated_image = image.copy()

    cv2.polylines(annotated_image, [mesh_points[LEFT_EYE]], True, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.polylines(annotated_image, [mesh_points[RIGHT_EYE]], True, (255, 0, 0), 1, cv2.LINE_AA)

    left_ear = calculate_EAR(mesh_points[left_eye_points])
    right_ear = calculate_EAR(mesh_points[right_eye_points])

    EAR = (left_ear + right_ear) / 2
    EAR = round(EAR, 2)
    if EAR < Threshold:
        drowsy = 'drowsy'
        cv2.putText(annotated_image, "DROWSY", (500, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        drowsy = 'not_drowsy'

    return (annotated_image, drowsy)

def alarm_on():
    sound.play(-1)

    current_time = datetime.datetime.now()
    message = "انذار لقد نام السائق"
    sql = "INSERT INTO alarm (time, message) VALUES (%s, %s)"
    val = (current_time.strftime('20%y-%m-%d %h %I.%M.%S'), message)
    mycursor.execute(sql, val)
    mydb.commit()

    with open('Database.csv', mode='a', newline='') as csvfile:
        csvfileWriter = csv.writer(csvfile)
        csvfileWriter.writerow([current_time.strftime('20%y-%m-%d %h %I.%M.%S'),"Warning"])

def alarm_off():
    sound.stop()
    pass


prev = 0
sw_started = False
font = cv2.FONT_HERSHEY_SIMPLEX
Threshold = 0.18 # the eye aspect ratio threshold
video_FPS = 25 # the frame rate of the recorded video
vid_duration = 60 # the duration of the recorded video
frame_rate = 25 # the frame rate of the camera caturing frames
video_start = time() # the moment the video start recording at
date = datetime.datetime.now() # the date of the moment this program starts
stop_watch = 0 # stop watch to calculate how long the driver closed his eyes
num_videos = 1000 # the maximum number of recorded video  to store in the flash
alarm_flag = False # flag to send only one alarm
delay_threashold = 1 # start the alarm if drowsy for {delay_threashold} seconds for example: 1 second
dir_videos = './videos' # the path where we stor videos in

if not isdir(dir_videos): mkdir(dir_videos)
videos = [f for f in listdir(dir_videos) if isfile(join(dir_videos, f))] # the list of the videos stored in the flash
path_vid = join(dir_videos, f'{date.year}-{date.month}-{date.day}_{date.hour}-{date.minute}.mp4') # the name of the video recorded


# Left eye indices list
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
left_eye_points = [362, 263, 384, 381, 386, 374, 387, 373]

# Right eye indices list
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
right_eye_points = [33, 133, 160, 144, 159, 145, 158, 153]


cap = cv2.VideoCapture(0) # the camera [capturer]
face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1) # the mediapipe model

_, frame = cap.read()
img_h, img_w, _ = frame.shape
size = (img_w, img_h)
video = cv2.VideoWriter(path_vid, cv2.VideoWriter_fourcc(*'mp4v'), video_FPS, size, 0) # the recorder




#------------- Start program
while cap.isOpened():

    time_elapsed = time() - prev
    frame = cv2.flip(cap.read()[1], 1)

    if time_elapsed > 1 / frame_rate:
        prev = time()
        annotated, drowsy = get_eyes_points(frame, face_mesh)

        if drowsy == 'drowsy':
            if not sw_started:
                stop_watch = prev
                sw_started = True
            sw_time = prev - stop_watch
            if sw_time > delay_threashold:
                cv2.putText(annotated, f'for {sw_time:.1f} sec',  (450, 110), font, 1, (100, 100, 100), 2)
                # start the alarm here:
                if not alarm_flag:
                    alarm_flag = True
                    alarm_on()

        else:
            sw_started = False
            if alarm_flag:
                alarm_flag = False
                alarm_off()

        current_time = datetime.datetime.now()
        cv2.putText(annotated, f'{current_time}',  (200, 35), font, 1, (100, 100, 100), 2)
        cv2.putText(annotated, f'FPS: {frame_rate}', (10, 25), font, 1, (20, 20, 20), 2)

        video.write(annotated)

        if time() - video_start > vid_duration:
            video_start = time()
            video.release()
            videos = [f for f in listdir(dir_videos) if isfile(join(dir_videos, f))]
            if len(videos) > num_videos:
                remove(join(dir_videos, videos[0]))
            date = datetime.datetime.now()
            path_vid = join(dir_videos, f'{date.year}-{date.month}-{date.day}_{date.hour}-{date.minute}.mp4')
            video = cv2.VideoWriter(path_vid, cv2.VideoWriter_fourcc(*'mp4v'), video_FPS, size, 0)

        cv2.imshow('Face Landmarks Detection', annotated)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

video.release()
cap.release()
cv2.destroyAllWindows()
