import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

input_path = r'/home/andrew/Vision Experiments/Lucy/'
output_path = r'/home/andrew/Vision Experiments/Lucy Faces/'

face_num = 0

base_output_name = 'lucy_{}.jpg'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for f_name in os.listdir(input_path):
    print('Reading {}'.format(f_name))
    img = cv2.imread(os.path.join(input_path, f_name))
    img = cv2.resize(img, (int(img.shape[1] * .25), int(img.shape[0] * .25)), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)

    print('   {} Faces detected'.format(len(faces)))

    for i in range(0, len(faces)):
        face = faces[0]
        x, y, w, h = face

        x2 = np.min((x+w, img.shape[1]))
        y2 = np.min((y+h, img.shape[0]))

        img_cropped = img[y:y2, x:x2]

        print('    Face {}, shape = {}'.format(i, img_cropped.shape))

        output_name = os.path.join(output_path, base_output_name.format(face_num))

        print('    Writing output to {}'.format(output_name))
#        plt.imsave(img_cropped, output_name)
        if cv2.imwrite(output_name, img_cropped):
           print(' ... successfully')
        face_num += 1
