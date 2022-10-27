from facial_library import FacialLibrary
import cv2
import os

face_lib = FacialLibrary(r'/home/andy/robotics/FaceRecognition/library')

log_dir = r'/home/andy/robotics/FaceRecognition/logging'

for f in os.listdir(log_dir):
    f_name = os.path.join(log_dir, f)
    print('Loading {}'.format(f_name))
    img = cv2.imread(f_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = face_lib.identify_face(img, nearest=True)
    cv2.imshow(face, img)
    cv2.waitKey(0)