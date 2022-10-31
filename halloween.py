import cv2
import keyboard
import pygame
import mediapipe as mp
import numpy as np
import time
import moviepy
from moviepy.editor import *
from cv2 import destroyAllWindows
from time import sleep
from threading import Thread
from playsound import playsound
from sys import exit

pygame.init()
pygame.mixer.init()
window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

look_up_time = 0
focus_time = 0
jump_scared = False

clips = []
clip = None

for i in range(137):
    if i + 1 < 10:
        clips.append(f"ezgif-frame-00{i+1}.jpg")
    elif i + 1 >= 10 and i + 1 < 100:
        clips.append(f"ezgif-frame-0{i+1}.jpg")
    else:
        clips.append(f"ezgif-frame-{i+1}.jpg")


def scream():
    pygame.mixer.music.load(
        "SCREAM.mp3")
    pygame.mixer.music.play()
    sleep(0.2)


def background_sound():
    pygame.mixer.music.load(
        "Scream Video.mp3")
    pygame.mixer.music.play(loops=1000)


def play_background():
    pygame.mixer.music.load(
        "Scream Video.mp3")
    pygame.mixer.music.play(loops=1000)
    image = pygame.image.load("Halloween.png")
    image_scaled = pygame.transform.scale(image, (1280, 800))
    window.blit(image_scaled, (0, 0))
    pygame.display.update()


def looking_up(up_time):
    print("This person is looking up.", up_time)
    time.sleep(1)
    up_time += 1
    return up_time


def looking(focus):
    print("Controlled Situation.", focus)
    time.sleep(1)
    focus += 1
    return focus


start_bgmusic_thread = Thread(
    target=play_background)

scream_thread = Thread(target=scream)

background_sound_thread = Thread(target=background_sound)


def after_jump_scare():
    # After Jumpscare
    global clips
    print("INFO: AFTER JUMPSCARE")
    background_sound_thread.start()
    print("INFO: BACKGROUND MUSIC STARTED")
    for i in range(10000):
        for clip in clips:
            image = pygame.image.load(f"Bloood/{clip}")
            image_scaled = pygame.transform.scale(image, (1280, 800))
            window.blit(image_scaled, (0, 0))
            pygame.display.update()


after_jump_scare_thread = Thread(target=after_jump_scare)


def jump_scare(status):
    print("INFO : JUMPSCARE INITIATED")
    scream_thread.start()
    global clips
    for clip in clips:
        image = pygame.image.load(f"Video Frame/{clip}")
        image_scaled = pygame.transform.scale(image, (1280, 720))
        window.blit(image_scaled, (0, 0))
        pygame.display.update()
    sleep(10)
    after_jump_scare_thread.start()


def override_jump_scare():
    while True:
        if keyboard.read_key() == "q":
            global look_up_time
            global jump_scared
            look_up_time = 100
            print("INFO: READY TO OVERRIDE PRESS w to OVERRIDE")

            while True:
                if keyboard.read_key() == "w":
                    jump_scared = True
                    scream_thread.start()
                    global clips
                    for clip in clips:
                        image = pygame.image.load(f"Video Frame/{clip}")
                        image_scaled = pygame.transform.scale(
                            image, (1280, 720))
                        window.blit(image_scaled, (0, 0))
                        pygame.display.update()
                    sleep(2)
                    after_jump_scare_thread.start()


stop_bgmusic_thread = Thread(target=jump_scare, args="J")

override_jump_scare_thread = Thread(target=override_jump_scare)

start_bgmusic_thread.start()
override_jump_scare_thread.start()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            focus_time = looking(focus_time)
            if y < -10:
                text = "Looking Left"
                look_up_time = looking_up(look_up_time)
                focus_time = 0
            elif y > 10:
                text = "Looking Right"
                look_up_time = looking_up(look_up_time)
                focus_time = 0
            elif x < -10:
                text = "Looking Left"
                look_up_time = looking_up(look_up_time)
                focus_time = 0
            elif x > 10:
                text = "Looking Up"
                look_up_time = looking_up(look_up_time)
                focus_time = 0
            else:
                text = "Forward"
                look_up_time = looking_up(look_up_time)
                focus_time = 0

            # Show jumpscare
            if look_up_time == 4:
                stop_bgmusic_thread.start()
                sleep(0.02)
                showPic = cv2.imwrite("filename.jpg", image)
                cv2.namedWindow("Window_name", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(
                    "Window_name", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            if jump_scared == True:
                sleep(0.02)
                showPic = cv2.imwrite("filename.jpg", image)

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start

        # fps = 1 / totalTime
        # print("FPS: ", fps)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Press U for override jumpscare
cap.release()
