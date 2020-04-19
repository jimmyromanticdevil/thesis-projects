#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import numpy as np
from sklearn.externals import joblib
from scipy.cluster.vq import *
from time import time
import utils_load as ut
import fsvm

# Load the classifier, class names, scaler, number of clusters and vocabulary
print ("======================================================")
print ("Membuat laporan teks yang menunjukkan metrik klasifikasi utama TEST\n")
# load with to see the prediction score using dataset
nyamuk_profile_data, nyamuk_profile_name_index, nyamuk_profile_names = ut.load_training_data(
    "data/train/")
print ("\n", nyamuk_profile_name_index.shape[0], " sample dari ", len(nyamuk_profile_names), " orang telah diload")

# membuat test data melalui buildSVC
build_test = fsvm.build_SVC(
    nyamuk_profile_data, nyamuk_profile_name_index, (500, 300), nyamuk_profile_names)

if build_test:
   print ("metrics.classification_report: SELESAI")
   
# load the bof file
clf, classes_names, standard_scaler, k, voc = joblib.load("bagoffeatures.pkl")

# Dapatkan path dari testing Set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
args = vars(parser.parse_args())

# Dapatkan path dari testing gambar dan taruh di dalam sebuah list
image_paths = []
t0 = time()
image_paths = [args["image"]]

# Proses membuat feature ekstraksi dan keypoint detektor obyek
# menggunakan SURF
fea_det = cv2.FeatureDetector_create("SURF")
des_ext = cv2.DescriptorExtractor_create("SURF")

# Daftar di mana semua deskriptor disimpan
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    if im is None:
        print ("No such file {}\nCheck if the file exists".format(image_path))
        exit()
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))

# Menumpuk semua deskriptor vertikal dalam array numpy
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))

# menghitung histogram fitur
features_test = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        features_test[i][w] += 1

# melakukan Tf-Idf vectorization
nbr_occurences = np.sum((features_test > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) /
                      (1.0 * nbr_occurences + 1)), 'float32')


features_test = standard_scaler.transform(features_test)

prob = clf._predict_proba_lr(features_test)
prob_per_class_dictionary = dict(zip(clf.classes_, prob[0]))
results_ordered_by_probability = map(lambda x: x[0], sorted(
    zip(clf.classes_, prob[0]), key=lambda x: x[1], reverse=True))
name = classes_names[results_ordered_by_probability[0]]

print ("======================================================")
print ("Memulai Prediksi Gambar: %s"%image_path)
print ("\nPrediksi dengan waktu %0.3fs" % (time() - t0))
print ("\nPrediksi Peluang: %s" % prob_per_class_dictionary[results_ordered_by_probability[0]])
print ("\nUrutan Prediksi: %s" % results_ordered_by_probability)
print ("\nList dan Urutan Prediksi:\n")
counter = 0
for i in results_ordered_by_probability:
    print("%s %s dengan presentase %s" % (counter, classes_names[i], prob_per_class_dictionary[i]))
    counter += 1


print ("\n\nHasil Prediksi : %s\n\n" % name)
print ("SELESAI")
print ("======================================================")
image = cv2.imread(image_path)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
for kp in kpts:
    x = int(kp.pt[0])
    y = int(kp.pt[1])
    cv2.circle(image, (x, y), 2, (0, 0, 255))

cv2.imshow("Image", image)
ch = 0xFF & cv2.waitKey() #tunggu program sampai esc di klick untuk keluar dari program
cv2.destroyAllWindows() #keluar dari program
