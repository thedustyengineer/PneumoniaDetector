import numpy as np
import cv2
from pdetect_perturbations import Perturbations as pb


def test_sp_noise():
    '''
    Call add_sp_noise on the test image with a large amount of noise and save the result, then do the same with a small amount of noise.
    '''
    test_img = cv2.imread('test_image.jpeg')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    if test_img is None:
        print('Image is None')
    high_noise_img = pb.add_sp_noise(test_img, 0.50, 0.01)
    cv2.imwrite('test_high_sp_noise.jpg',high_noise_img)
    low_noise_img = pb.add_sp_noise(test_img, 0.50, 0.00001)
    cv2.imwrite('test_low_sp_noise.jpg',low_noise_img)

def test_gauss_noise():
    '''
    Call add_gauss_noise on the test image with a large amount of noise and save the result, then do the same with a small amount of noise.
    '''
    test_img = cv2.imread('test_image.jpeg')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    if test_img is None:
        print('Image is None')
    high_noise_img = pb.add_gauss_noise(test_img,10)
    cv2.imwrite('test_high_gauss_noise.jpg',high_noise_img)
    low_noise_img = pb.add_gauss_noise(test_img, 1)
    cv2.imwrite('test_low_gauss_noise.jpg',low_noise_img)


test_sp_noise()
test_gauss_noise()