import dlib
import cv2
import face_recognition
import time
from facial_library import FacialLibrary
from jetson_utils import videoSource, videoOutput, cudaFromNumpy, cudaToNumpy

dlib.DLIB_USE_CUDA = True

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture(0)
cap = videoSource()
output = videoOutput()

method = 'fr_CNN'
#method = 'cv2'
#method = 'fr_HOG'

start_time = time.time()
num_frames = 0

scale_factor = 0.25

face_lib = FacialLibrary(r'/home/andy/robotics/FaceRecognition/library')

while True:
    num_frames += 1

    #ret, frame = cap.read()
    cuda_img = cap.Capture()
    frame = cudaToNumpy(cuda_img)

    frame_small = cv2.resize(frame, (int(frame.shape[1] * scale_factor), 
                               int(frame.shape[0] * scale_factor)),
                            interpolation=cv2.INTER_AREA)
    frame_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)

    people_found = face_lib.identify_faces(frame_small, nearest=True)

    for the_name, (x1, y1, x2, y2) in people_found:
        if len(the_name) > 0:
            x1 = int(x1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y1 = int(y1 / scale_factor)
            y2 = int(y2 / scale_factor)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame,
                        the_name,
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        1,
                        2)

    cuda_img = cudaFromNumpy(frame)

    output.Render(cuda_img)
    #output.SetStatus("FPS {} Using CUDA = {}".format(round(num_frames / (time.time() - start_time), 2)))
    output.SetStatus("FPS {} Using CUDA = {}".format(round(num_frames / (time.time() - start_time), 2), 
                                                     dlib.DLIB_USE_CUDA))


    # exit on input/output EOS
    if not cap.IsStreaming() or not output.IsStreaming():
        break

    # frame = cudaToNumpy(cuda_img)
    # 

    #print(type(frame))

    

    #         print('Found: {}'.format(the_name))


    #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4)

    # if 'fr' in method:
    #     model = method.split('_')[1]
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     faces = face_recognition.face_locations(frame_rgb, model=model)
    # else:
    #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4)
    #
    # print('{} Faces detected (FPS: {})'.format(len(faces), round(num_frames / (time.time() - start_time))))

    # if len(faces) > 0:
    #     face = faces[0]
    #     x, y, w, h = face
    #     frame = frame[y:y+h,x:x+w]
    #     cv2.imshow("Frame", frame)

    # Draw rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # if 'fr' in method:
    #     for (y, x2, y2, x) in faces:
    #         cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
    # else:
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #cv2.imshow(method, frame)
    # cudaOut = cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # output.Render(cudaOut)


# img = cv2.imread(r'/home/andrew/Vision Experiments/Kerry/2021-06-16_11-30-05_432.jpg')
# print('Original Dimensions: ', img.shape)
#
# img = cv2.resize(img, (int(img.shape[1] * .25), int(img.shape[0] * .25)), interpolation=cv2.INTER_AREA)
# print('Resized Dimensions: ', img.shape)
#
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# img_encoding = face_recognition.face_encodings(img_rgb)
#
#
#
# cv2.imshow("Kerry", img)
# cv2.waitKey(0)
