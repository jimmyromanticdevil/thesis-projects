#!/usr/local/bin/python2.7
"""
Susunan Kode ini bertugas untuk mengekstrak Citra pada gambar
menggunakan metode SURF, K-Means dan SVM dan akan menghasilkan Bag Of Feature 

"""
import argparse as ap
import cv2
import utils
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from time import time

t0 = time()
print "Mendeteksi dan Mengekstraksi Feature..."
# Dapatkan lokasi folder dari training set
parser = ap.ArgumentParser()
parser.add_argument("-l", "--latih",
                    help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# dapatkan nama class dari training dan store mereka dalam sebuah list
train_path = args["latih"]
training_names = os.listdir(train_path)

# dapatkan semua lokasi utk gambar dan menaruh di dalam list image_paths
# serta nama label folder
# misal nama folder albo_female, aegpyte_male dll
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = utils.imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# Proses membuat feature ekstraksi dan keypoint detektor obyek
# menggunakan SURF
fea_det = cv2.FeatureDetector_create("SURF")
des_ext = cv2.DescriptorExtractor_create("SURF")

# Daftar di mana semua deskriptor disimpan
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))

# Menumpuk semua deskriptor vertikal dalam array numpy
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1)

# menghitung histogram fitur
im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) /
                      (1.0 * nbr_occurences + 1)), 'float32')

# Scaling kata-kata
print "Proses Descriptor..."
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Training Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Simpan hasil Training
joblib.dump((clf, training_names, stdSlr, k, voc),
            "bagoffeatures.pkl", compress=3)
print "Selesai"
print "Prediksi dengan waktu %0.3fs" % (time() - t0)
