import dlib
import cv2
import face_recognition
import time
from FaceRecognition.facial_library import FacialLibrary
from jetson_utils import videoSource, videoOutput, cudaFromNumpy, cudaToNumpy, cudaFont, cudaDrawRect
from jetson_inference import poseNet
import argparse
import sys
import numpy as np
import math

from Control.CameraController import CameraController

def get_target_point(pose, which_target):
    if which_target == "eyes":
        left_eye_idx = pose.FindKeypoint('left_eye')
        right_eye_idx = pose.FindKeypoint('right_eye')

        if left_eye_idx >= 0 and right_eye_idx >= 0:
            # Found two eyes in the image
            left_eye = pose.Keypoints[left_eye_idx]
            right_eye = pose.Keypoints[right_eye_idx]

            mid_x = (left_eye.x + right_eye.x) / 2
            mid_y = (left_eye.y + right_eye.y) / 2
        
            return (mid_x, mid_y)
    else:
        idx = pose.FindKeypoint(which_target)

        if idx >= 0:
            pt = pose.Keypoints[idx]
            return (pt.x, pt.y)

    return (-1, -1)

dlib.DLIB_USE_CUDA = True

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture(0)
input = videoSource()
output = videoOutput()

method = 'fr_CNN'
#method = 'cv2'
#method = 'fr_HOG'

do_face_recognition = True
do_face_track = True

pan_full = 25
tilt_full = 15
look_tol = 50

track_target = "eyes"

identify_counter = 20

scale_factor = 0.25

camera_controller = CameraController()
camera_controller.update()

face_lib = FacialLibrary(   r'/home/andy/robotics/FaceRecognition/library', 
                            tracking_tol=100,
                            log_face_dir=r'/home/andy/robotics/FaceRecognition/logging')

font = cudaFont()

parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

net = poseNet(opt.network, sys.argv, opt.threshold)

face = ""

start_time = 0
num_frames = 0

while True:
    num_frames += 1

    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    #poses = net.Process(img, overlay=opt.overlay)
    poses = net.Process(img, overlay="none")

    if start_time == 0:
        start_time = time.time()

    # print the pose results
    #print("detected {:d} objects in image".format(len(poses)))

    if len(poses) == 0:
        face = ""

    found_faces = []

    for pose in poses:
        # print(pose)
        # print(pose.Keypoints)
        # print('Links', pose.Links)

        if do_face_track:
            mid_x, mid_y = get_target_point(pose, track_target)

            if mid_x >= 0 and mid_y >= 0:
                center_x = img.width / 2
                center_y = img.height / 2

                print("Center ({}, {})".format(round(center_x), round(center_y)))
                print("Eye    ({}, {})".format(round(mid_x), round(mid_y)))
                
                x_dist = np.abs(center_x - mid_x)
                y_dist = np.abs(center_y - mid_y)

                x_degrees = int(x_dist / center_x * pan_full)
                y_degrees = int(y_dist / center_y * tilt_full)

                print('Panning {}, Tilting {}'.format(x_degrees, y_degrees))

                if center_x - mid_x > look_tol:
                    camera_controller.PanLeft(degrees=x_degrees, do_update=False)
                elif center_x - mid_x < -look_tol:
                    camera_controller.PanRight(degrees=x_degrees, do_update=False)

                if center_y - mid_y > look_tol:
                    camera_controller.TiltUp(degrees=y_degrees, do_update=False)
                elif center_y - mid_y < -look_tol:
                    camera_controller.TiltDown(degrees=y_degrees, do_update=False)

                camera_controller.update()
                camera_controller.dump()

        if do_face_recognition:

            left_eye_idx = pose.FindKeypoint('left_eye')
            right_eye_idx = pose.FindKeypoint('right_eye')

            if left_eye_idx >= 0 and right_eye_idx >= 0:
                # Found two eyes in the image
                left_eye = pose.Keypoints[left_eye_idx]
                right_eye = pose.Keypoints[right_eye_idx]

                mid_x = (left_eye.x + right_eye.x) / 2
                mid_y = (left_eye.y + right_eye.y) / 2

                face = face_lib.check_recent_face(mid_x, mid_y)
            
                eye_w = np.abs(left_eye.x - right_eye.x)
                eye_h = eye_w * 2

                # Eyes are from the perspective of the person, not the viewer
                # Left eye is on the right side of the screen if the person is looking at you...
                x1 = int(right_eye.x - eye_w)
                x2 = int(left_eye.x + eye_w)
                y1 = int(np.min((right_eye.y, left_eye.y)) - eye_h)
                y2 = int(np.max((right_eye.y, left_eye.y)) + eye_h)

                # Do Facial Recognition
                if face == "": #or num_frames % identify_counter == 0:
                    # print('Left Eye: ({}, {})'.format(round(left_eye.x), round(left_eye.y)))
                    # print('Right Eye: ({}, {})'.format(round(right_eye.x), round(right_eye.y))
                    # print('Face Coords: ({}, {}), ({}, {})'.format(x1, y1, x2, y2))

                    img_np = cudaToNumpy(img)

                    # Crop down to the face
                    img_np = img_np[y1:y2, x1:x2, :]

                    # Reduce size to speed up inference (do we need this?)
                    img_np_small = img_np
                    # img_np_small = cv2.resize(img_np, (int(img_np.shape[1] * scale_factor), 
                    #                         int(img_np.shape[0] * scale_factor)),
                    #                         interpolation=cv2.INTER_AREA)

                    # Don't need this since .identify_face expects RGB already
                    #img_np_small = cv2.cvtColor(img_np_small, cv2.COLOR_RGB2BGR)

                    face = face_lib.identify_face(img_np_small, nearest=True)

                    face_lib.track_face(face, mid_x, mid_y)

                if face != "":
                    cudaDrawRect(img, (x1, y1, x2, y2), (255, 0, 0, 50))
                    font.OverlayText(img, img.width, img.height, face, int(right_eye.x), np.max((0, int(right_eye.y-100))), font.White, font.Gray40)
                    found_faces.append(face)

    face_lib.keep_faces(found_faces)

    # render the image
    output.Render(img)

    # update the title bar
    #output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
    output.SetStatus("FPS {}".format(round(num_frames / (time.time() - start_time), 1)))

    # num_frames += 1

    # #ret, frame = cap.read()
    # cuda_img = cap.Capture()
    # frame = cudaToNumpy(cuda_img)

    # frame_small = cv2.resize(frame, (int(frame.shape[1] * scale_factor), 
    #                            int(frame.shape[0] * scale_factor)),
    #                         interpolation=cv2.INTER_AREA)
    # frame_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)

    # people_found = face_lib.identify_faces(frame_small, nearest=True)

    # for the_name, (x1, y1, x2, y2) in people_found:
    #     if len(the_name) > 0:
    #         x1 = int(x1 / scale_factor)
    #         x2 = int(x2 / scale_factor)
    #         y1 = int(y1 / scale_factor)
    #         y2 = int(y2 / scale_factor)
            
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #         cv2.putText(frame,
    #                     the_name,
    #                     (x1, y1),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     1,
    #                     (255, 0, 0),
    #                     1,
    #                     2)

    # cuda_img = cudaFromNumpy(frame)

    # output.Render(cuda_img)
    # #output.SetStatus("FPS {} Using CUDA = {}".format(round(num_frames / (time.time() - start_time), 2)))
    # output.SetStatus("FPS {} Using CUDA = {}".format(round(num_frames / (time.time() - start_time), 2), 
    #                                                  dlib.DLIB_USE_CUDA))


    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
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
