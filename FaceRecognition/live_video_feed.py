import cv2
import face_recognition
import time
from facial_library import FacialLibrary

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

method = 'fr_CNN'
#method = 'cv2'
#method = 'fr_HOG'

start_time = time.time()
num_frames = 0

face_lib = FacialLibrary(r'/home/andrew/Vision Experiments/All_2')

while True:
    num_frames += 1

    ret, frame = cap.read()

    people_found = face_lib.identify_faces(frame)

    for the_name, (x1, y1, x2, y2) in people_found:
        if len(the_name) > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame,
                        the_name,
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        1,
                        2)

            print('Found: {}'.format(the_name))


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

    cv2.imshow(method, frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


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
