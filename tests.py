import numpy as np
import cv2
from pdetect_perturbations import Perturbations as pb

def test_pad_image():
    '''
    Call pad_image on the test image to pad an image and save the results.
    '''
    test_img = cv2.imread('test_image.jpeg')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    if test_img is None:
        print('Image is None')
    test_img = pb.pad_image(test_img, 50, 50)
    cv2.imwrite('padded_img.jpeg', test_img)

def test_rotate_image():
    '''
    Call rotate_image on the test image and rotate the image 5 times and save the results.
    '''
    test_img = cv2.imread('test_image.jpeg')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    if test_img is None:
        print('Image is None')
    degrees = [np.random.uniform(-5, 5) for i in range(5)]
    for i in degrees:
        test_img = pb.rotate_image(test_img, i)
        cv2.imwrite('rotated_img_' + str(i) +'.jpeg', test_img)

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


def test_adjust_brightness():
    '''
    Call adjust_brightness on the test image and save the result to file
    '''
    test_img = cv2.imread('test_image.jpeg')
    if test_img is None:
        print('Image is None')
    img_file_index = 0    
    for i in range(-10, 10, 3):
        brightned_img = pb.adjust_brightness(test_img, i)
        img_file_index = img_file_index + 1
        cv2.imwrite('test_brightned_img' + str(img_file_index) + '.jpg', brightned_img)

def test_mirror_image():
    '''
    Call mirror_image on the test image and save the result to file
    '''
    test_img = cv2.imread('test_image.jpeg')
    mirrored_image = pb.mirror_image(test_img)
    cv2.imwrite('mirrored_image' + '.jpg', mirrored_image)

def test_resize_image():
    '''
    Call resize_image on the test image then pad to 256 x 256 and save the result to file
    '''
    test_img = cv2.imread('test_image.jpeg')
    mirrored_image = pb.resize_image(test_img, 256)
    mirrored_image = pb.pad_image(mirrored_image, 256, 256)
    cv2.imwrite('mirrored_image' + '.jpg', mirrored_image)

#test_adjust_brightness()
#test_rotate_image()
#test_sp_noise()
#test_gauss_noise()
#test_pad_image()
#test_mirror_image()
test_resize_image()