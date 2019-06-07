import face_recognition
import cv2
# objective: this code will help you in running face recognition on a video file and saving the results to a new video file.

# ---init variables----
input_video_name = "front_original.mp4"
output_video_name = "front_recognition.avi"

# make sure resolution/frame rate matches input video!
fps = 29.97
frame_size = (1280,720)

known_faces_img = ["jfla3.png","jfla2.jpg"]
known_faces_name = ["jfla","jfla2"]
# ---------------------

# "VideoCapture" is a class for video capturing from video files, image sequences or cameras
input_video = cv2.VideoCapture(input_video_name)
#"CAP_PROP_FRAME_COUNT": it helps in finding number of frames in the video file.
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
# Create an output movie file
#  So we capture a video, process it frame-by-frame and we want to save that video, it only possible by using "VideoWriter" object
# FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org. It is platform dependent.
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # DVIX is for the windows environment
# number of frames per second (fps), frame size
output_video = cv2.VideoWriter(output_video_name, fourcc, fps, frame_size)
# Load some sample pictures and learn how to recognize them.
known_faces = []
for img in known_faces_img:
    known_img = face_recognition.load_image_file(img)
    known_face_encoding = face_recognition.face_encodings(known_img)[0]
    known_faces.append(known_face_encoding)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0
while True:
    # Grab a single frame of video
    ret, frame = input_video.read()
    frame_number += 1
# Quit when the input video file ends
    if not ret:
        break
# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
# Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        for i,v in enumerate(known_faces_name):
            if match[i]:
                name = v
    face_names.append(name)
# Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
# Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
# Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
# Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_video.write(frame)
# All done!
input_video.release()
cv2.destroyAllWindows()