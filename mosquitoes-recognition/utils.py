#!/usr/local/bin/python2.7

import cv2
import os
import matplotlib.pyplot as plt


def imlist(path):
    """
    fungsi dari imlist mengembalikan semua nama file di dalam direktori

    """
    return [os.path.join(path, f) for f in os.listdir(path)]


def imshow(im_title, im):
    '''
    Fungsi ini di gunakan untuk menampilkan gambar ke layar
    '''
    plt.figure()
    plt.title(im_title)
    plt.axis("off")
    if len(im.shape) == 2:
        plt.imshow(im, cmap="gray")
    else:
        im_display = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        plt.imshow(im_display)
    plt.show()


def imreads(path):
    ''' 
    fungsi ini membaca semua gambar di dalam sebuah folder dan mengembalikan
    hasilnya
    '''
    paths = os.path.dirname(os.path.abspath(__file__))
    images_path = imlist(paths)
    images = []
    for image_path in images_path:
        images.append(cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR))
    return images


def show(image, name="Image"):
    '''
    Menampilkan gambar secara rutin
    '''
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
