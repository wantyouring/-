import cv2
import dlib

# 얼굴 추출해 저장하는 코드
detector = dlib.get_frontal_face_detector()
file_name = "jfla.jpg" # 파일명 입력

img = cv2.imread(file_name)
dets = detector(img, 1)
for i, d in enumerate(dets):
    print("Detection{} : {}".format(i,d))
    crop = img[d.top():d.bottom(), d.left():d.right()]
    cv2.imwrite("{}{}.png".format(file_name.split(".")[0],i), crop)