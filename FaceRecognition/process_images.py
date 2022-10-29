from facial_library import FacialLibrary
import pickle
import cv2
import numpy as np

face_lib = FacialLibrary(r'/home/andrew/Vision Experiments/All_Balanced')
#face_lib.process_train_set()
face_lib.cluster_embeddings(threshold=0.46)

# thresh = np.arange(0.3, 0.50, 0.01)
#
# for t in thresh:
#     labels = face_lib.count_clusters(threshold=t)
#     print(round(t, 2), labels)

#
# embeddings = pickle.load(open(r'/home/andrew/Vision Experiments/All_Balanced/all_embeddings.pkl', 'rb'))
#
# for face in embeddings:
#     face_lib.display_face(face, "Test")
#     cv2.waitKey(0)