from __future__ import print_function
import sys
import cv2
import os

path = "tampung/hasil.jpg"
def main(argv):
    cap = cv2.VideoCapture(1)
    cap.set(3, 1920)
    cap.set(4, 1080)
    while True:
        ret, img = cap.read()
        cv2.imshow("input", img)
        key = cv2.waitKey(10)
        if key == 27:
            break
        elif key == 32:
            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            #cv2.imwrite("opencv.png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 90])
            os.system("python main_program.py %s"%path)

    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()

if __name__ == '__main__':
    main(sys.argv)
