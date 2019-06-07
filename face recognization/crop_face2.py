import face_recognition
import cv2

image = face_recognition.load_image_file("img.jpg")
face_locations = face_recognition.face_locations(image) # 얼굴 위치들 찾기
face_landmarks_list = face_recognition.face_landmarks(image) # 눈, 코, 입, 턱 윤곽 잡음

print(face_locations)
print(face_landmarks_list)