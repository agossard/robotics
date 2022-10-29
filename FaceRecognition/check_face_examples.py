from facial_library import FacialLibrary
import cv2
import os

face_lib = FacialLibrary(r'/home/andrew/PycharmProjects/robotics/FaceRecognition/library')

#log_dir = r'/home/andrew/PycharmProjects/robotics/FaceRecognition/logging'
log_dir = r'/home/andrew/Vision Experiments/All_2/clusters/test'

for f in os.listdir(log_dir):
    f_name = os.path.join(log_dir, f)
    print('Loading {}'.format(f_name))
    img = cv2.imread(f_name)
    print('Image Size ({}, {})'.format(img.shape[0], img.shape[1]))

    w = img.shape[1]
    h = img.shape[0]

    img = img[-w:,:]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = face_lib.identify_face(img_rgb, nearest=True)
    cv2.imshow(face, img)
    cv2.waitKey(0)