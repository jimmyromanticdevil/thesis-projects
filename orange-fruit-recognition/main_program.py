import sys
import cv2
import math
import numpy
from scipy.ndimage import label
pi_4 = 4*math.pi

"""
pertama tama kita harus memastikan tidak ada overlapping dalam gambar. lalu dapatkan edge dalam gambar secarah mudah
binarizing respon dari edge detector denga Otsu method https://en.wikipedia.org/wiki/Otsu%27s_method, filling holes dan terakhir adalah mengukur jumlah bundaran. dan jika masih terdapat overlaps pada gambar. kita bisa menggunakan watershed transform dengan di kombinasikan dengan distance transform untuk memisahkan droplets(titis kecil). masalahnya adalah kita tidak bisa mendapatkan semua titik. tapi sudah sangat cukup untuk mengukur dan membandingkan antara jeruk merah dan putih. dengan titik2 yang terdeteksi. jika titik sedikit itu karena putih menghasilkan titik yang sedikit terdeteksi karena permukaan/teksture putih tidak kasar layaknya jeruk merah
"""


def distance_segment(img):
    border = img - cv2.erode(img, None)
    #distance transform
    dt = cv2.distanceTransform(255 - img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)

    lbl, ncc = label(dt)
    lbl[border == 255] = ncc + 1

    #watershed algorthm 
    lbl = lbl.astype(numpy.int32)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), lbl)
    lbl[lbl < 1] = 0
    lbl[lbl > ncc] = 0

    lbl = lbl.astype(numpy.uint8)
    lbl = cv2.erode(lbl, None)
    lbl[lbl != 0] = 255
    return lbl


def find_texture(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #Gaussian Blur metode enhancement agar gambar mudah di proses.
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 2)

    #Erode atau Eroding agar gambar mudah di proses.
    edges = frame_gray - cv2.erode(frame_gray, None)

    #Otshu Threshold
    _, bin_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_OTSU)

    #filling hole algorthm
    height, width = bin_edge.shape
    mask = numpy.zeros((height+2, width+2), dtype=numpy.uint8)
    cv2.floodFill(bin_edge, mask, (0, 0), 255)

    #Watershed Algorthm
    components = distance_segment(bin_edge)

    #Extract semua point dan gabungkan jadi satu untuk mencari Contour (Garis Contour)
    
    circles, obj_center = [], []
    contours, _ = cv2.findContours(components,
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        c = c.astype(numpy.int64)
        area = cv2.contourArea(c)
        if 100 < area < 3000:
            arclen = cv2.arcLength(c, True)
            circularity = (pi_4 * area) / (arclen * arclen)
            if circularity > 0.5:
                circles.append(c)
                box = cv2.boundingRect(c)
                obj_center.append((box[0] + (box[2] / 2), box[1] + (box[3] / 2)))

    return circles, obj_center

if __name__ == "__main__":
   #menjalankan program dengan parameter argv(Argument passing)
   #python main_program.py argument
   if len(sys.argv) <= 1:
      #jika argument kurang dari 1 maka program tidak terpenuhi untuk bisa berjalan
      print "cara menjalankan"
      print "python main_program.py pathfile"
      sys.exit()

   #tampung path image/image di variabel ini. dengan mengambil index argument passingnya 
   img_file = sys.argv[1]       

   #baca file image dan tampung di variabel untuk segera di proses
   frame = cv2.imread(img_file)

   #lempar ke fungsi find_texture untuk segera di proses. dan tampung hasil di variabe circle
   circles, new_center = find_texture(frame)
   found = 'putih.jpg'

   #variabel circle menampung titik/ jumlah pola yang terdeteksi. dari hasil eksperiment di atas 40 titik/pola maka buah tersebut merah
   #sebalikna di bawah 40 buah tersebut adalah putih. karena texture pola putih sangat berbeda dgn merah.

   if len(circles) >= 40:
      found = 'merah.jpg'

   #buat window untuk tampilkan hasil gambar
   cv2.namedWindow('Hasil', cv2.WINDOW_NORMAL)

   #baca hasil gambar yang sudah terdeteksi. jika merah akan di panggil gambar buah merah. jika putih akan di panggil gambar buah putih 
   result_frame = cv2.imread(found) #meload gambar

   print "Hasilnya adalah: %s"% found.replace(".jpg","")
   cv2.imshow('Hasil', result_frame) #tampilkan gambar

   ch = 0xFF & cv2.waitKey() #tunggu program sampai esc di klick untuk keluar dari program
   cv2.destroyAllWindows() #keluar dari program

