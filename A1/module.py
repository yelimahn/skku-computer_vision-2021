import cv2
import numpy as np
import time

def showsave_image(img, filename):
    cv2.imshow(f"{filename}.png", (img).astype(np.uint8))
    cv2.waitKey()
    cv2.imwrite(f"result./{filename}.png", (img).astype(np.uint8))

def time_start():
    return time.time()

def time_end(start):
    return time.time() - start

def load_image(filename):
    return cv2.imread(f'{filename}.png', cv2.IMREAD_GRAYSCALE)