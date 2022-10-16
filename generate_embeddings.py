import cv2
import os
import numpy as np
import face_recognition
from matplotlib import pyplot as plt
import time
import pickle
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import math

encoding_method = 'CNN'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

input_path = r'/home/andrew/Vision Experiments/All_2/train'
output_file = r'/home/andrew/Vision Experiments/All_2/embeddings.pkl'
output_dir = r'/home/andrew/Vision Experiments/All_2/clusters'

def detect_faces(img_raw, method):

    faces_out = []

    if 'fr' in method:
        model = method.split('_')[1]
        frame_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(frame_rgb, model=model)

        for (y, x2, y2, x) in face_locs:
            faces_out.append((x, y, x2, y2))
    else:
        frame_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        face_locs = face_cascade.detectMultiScale(frame_gray, 1.1, 4)

        for (x, y, w, h) in face_locs:
            faces_out.append((x, y, x+w, y+h))

    return faces_out


def examine_faces(scale_factor=0.25):
    embeddings = pickle.load(open(output_file, 'rb'))

    for face in embeddings:
        display_face(face)
        cv2.waitKey(0)

def get_face(face, scale_factor=0.25):
    img = cv2.imread(os.path.join(input_path, face['f_name']))
    img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)),
                     interpolation=cv2.INTER_AREA)

    x, y, x2, y2 = face['loc']
    return img[y:y2,x:x2,:]

def display_face(face, caption="face", scale_factor=0.25):
    img = cv2.imread(os.path.join(input_path, face['f_name']))
    img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)),
                     interpolation=cv2.INTER_AREA)

    x, y, x2, y2 = face['loc']

    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)
    cv2.imshow(caption, img)

def compare_faces(scale_factor=0.25, threshold=0.5):
    embeddings = pickle.load(open(output_file, 'rb'))

    target_face = embeddings[0]
    display_face(target_face, "Target", scale_factor)

    cv2.waitKey(0)

    for face in embeddings:
        distance = math.dist(target_face['embedding'], face['embedding'])
        display_face(face, 'Distance: {}'.format(round(distance, 2)))
        cv2.waitKey(0)

def generate_embeddings(method='cv2', scale_factor=0.25):

    embeddings = []

    num_faces = 0
    num_embeddings = 0

    start_time = time.time()
    file_names = os.listdir(input_path)
    for i, f_name in enumerate(file_names):
        img = cv2.imread(os.path.join(input_path, f_name))

        if scale_factor < 1:
            img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

        faces = detect_faces(img, method)

        total_time = time.time() - start_time

        for face in faces:
            x1, y1, x2, y2 = face

            img_cropped = img[y1:y2, x1:x2]

            # cv2.imshow('Face', img_cropped)
            # cv2.waitKey(0)

            if 'fr' in method:
                img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

            emb = face_recognition.face_encodings(img_cropped,
                                                  known_face_locations=[(0, x2-x1, y2-y1, 0)],
                                                  model=encoding_method)
            emb = emb[0]

            if len(emb) > 0 and len(np.where(emb == 0)[0]) == 0:
                num_embeddings += 1

            num_faces += 1

            embeddings.append({'f_name': f_name, 'loc': face, 'embedding': emb})

        print('Read {} of {} files, {} faces, {} embeddings ({} faces per second, {} files per second)'.format(i,
                                                                                                                len(file_names),
                                                                                                                num_faces,
                                                                                                                num_embeddings,
                                                                                                                round(num_faces / total_time, 1),
                                                                                                                round(i / total_time, 1)))

    pickle.dump(embeddings, open(output_file, 'wb'))

def load_distances():
    embeddings = pickle.load(open(output_file, 'rb'))

    dim = len(embeddings[0]['embedding'])
    n = len(embeddings)

    x = np.ndarray(shape=(n,dim))

    for i, emb in enumerate(embeddings):
        x[i, :] = emb['embedding']

    return x

def cluster_embeddings(output_dir, x=None, threshold=0.42, scale_factor=0.25):
    x = load_distances() if x is None else x

    embeddings = pickle.load(open(output_file, 'rb'))

    dbs = DBSCAN(eps=threshold, metric='euclidean', n_jobs=-1)
    dbs.fit(x)

    n_labels = np.max(dbs.labels_) + 1

    for cluster in range(0, n_labels):

        face_ids = np.where(dbs.labels_ == cluster)[0]
        print('Writing cluster {} with {} faces'.format(cluster, len(face_ids)))

        cluster_dir = os.path.join(output_dir, 'cluster_{}'.format(cluster))
        os.makedirs(cluster_dir)

        for id in face_ids:
            f_name = os.path.join(cluster_dir, 'face_{}.jpg'.format(id))
            print('   writing: {}'.format(f_name))
            face = get_face(embeddings[id], scale_factor=scale_factor)
            cv2.imwrite(f_name, face)

def count_clusters(x=None, threshold=0.42):
    x = load_distances() if x is None else x

    dbs = DBSCAN(eps=threshold, metric='euclidean', n_jobs=-1)
    dbs.fit(x)

    labels = np.arange(0, 6)
    label_counts = [len(np.where(dbs.labels_ == l)[0]) for l in labels]
    return np.array(label_counts)

def examine_distances(x=None):
    x = load_distances() if x is None else x

    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.grid()
    plt.title('Distance to nearest neighbor')

def dbscan_hyperparams():
    thresholds = np.arange(0.1, 1.0, 0.01)
    results = []
    for t in thresholds:
        print('Running {}'.format(t))
        results.append(np.expand_dims(cluster_embeddings(threshold=t), axis=1).T)

    results = np.concatenate(results, axis=0)

    plt.plot(thresholds, results[:,0:2])
    plt.grid()
    plt.title('1st and 2nd Cluster Membership')
    plt.xlabel('EPS Parameter')


#examine_distances()
#compare_faces()
cluster_embeddings(threshold=0.42, output_dir=output_dir)
#examine_faces()
#generate_embeddings(method='fr_CNN', scale_factor=0.25)

print('finished')
