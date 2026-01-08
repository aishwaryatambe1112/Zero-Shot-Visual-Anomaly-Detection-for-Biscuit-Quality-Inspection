import cv2
import numpy as np

def check_burnt(biscuit_img):
    hsv = cv2.cvtColor(biscuit_img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    dark_pixels = np.sum(v < 60)
    total_pixels = v.size

    burn_ratio = dark_pixels / total_pixels

    if burn_ratio > 0.18:
        return True, burn_ratio

    return False, burn_ratio

