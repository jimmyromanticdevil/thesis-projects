def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from time import time
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split


def build_SVC(nyamuk_profile_data, nyamuk_profile_name_index, nyamuk_dim, nyamuk_profile_names):
    """
    Pengujian: Membangun SVM classification model dengan menggunakan matrix nyamuk_profile_data (numOfFace X numOfPixel) dan nyamuk_profile_name_index array, nyamuk_dim is a tuple of the dimension of each image(h,w) Returns the SVM classification modle
    Parameters
    ----------
    nyamuk_profile_data : ndarray (number_of_images_in_nyamuk_profiles, width * height of the image)

    nyamuk_profile_name_index : ndarray
        Nama yang sesuai dengan profil nyamuk dikodekan dalam indeks

    nyamuk_dim : tuple (int, int)
        The dimension of the face data is reshaped to
        Dimensi dari data nyamuk yang telagh dibentuk kembali

    nyamuk_profile_names: ndarray
        Nama-nama yang sesuai dengan profil nyamuk

    Returns
    -------
    clf : theano object
        jenis model klasifikasi SVM yang dilatih

    """
    t0 = time()
    X = nyamuk_profile_data
    y = nyamuk_profile_name_index

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    # Menghitung sebuah PCA (eigenfaces) pada dataset nyamuk
    # (diperlakukan sebagai dataset berlabel):
    # unsupervised  ekstraksi fitur / pengurangan dimensi
    n_components = 150  # maximum number of components to keep
    print("\nMengekstrak %d dataset dari %d nyamuk" %
          (n_components, X_train.shape[0]))
    print("\nMemproyeksikan input data atas dasar ortonormal")

    # Melatih model klasifikasi SVM
    print("\nMencocokkan classifier ke training set")
    # Perkiraan terbaik yang ditemukan menggunakan Radial Basis Function Kernal:
    # Scaling kata-kata
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Training Linear SVM
    clf = LinearSVC()
    clf = clf.fit(X_train, y_train)
    """

    clf = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC())])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    ##########################################################################
    # evaluasi kuantitatif dari kualitas model pada uji set
    print("\nMemprediksi tipe nyamuk pada test set")
    print("\nPrediksi mengambil %s per sample dari rata-rata" %
          ((time() - t0) / y_pred.shape[0] * 1.0))
    print("selesai dalam waktu %0.3fs" % (time() - t0))
    print("\n")
    print "Laporan klasifikasi untuk classifier %s:\n%s\n" % (
        clf, metrics.classification_report(y_test, y_pred, target_names=nyamuk_profile_names))
    print "Accuracy metrics %s:\n" % metrics.accuracy_score(y_test, y_pred)

    error_rate = errorRate(y_pred, y_test)
    print("\nUji Tingkat Kesalahan: %0.4f %%" % (error_rate * 100))
    print("Uji Tingkat Rekognisi: %0.4f %%" % ((1.0 - error_rate) * 100))

    return True


def errorRate(pred, actual):
    """
    Mengkalkulasi nama prediksi error rate

    Parameter
    ----------
    pred: ndarray (1, number_of_images_in_nyamuk_profiles)
        Nama-nama yang diprediksi dari dataset uji


    actual: ndarray (1, number_of_images_in_nyamuk_profiles)
        Nama sebenarnya dari dataset uji

    Return
    -------
    error_rate: float
        tingkat kesalahan yang dihitung

    """
    if pred.shape != actual.shape:
        return None
    error_rate = np.count_nonzero(pred - actual) / float(pred.shape[0])
    return error_rate
