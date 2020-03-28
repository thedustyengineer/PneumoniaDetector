import cv2
import random

def rotate_image(image, rot_val):
    rot_val = random.uniform(-5, 5)

    image = cv2.imread('lena.jpg', 0)
    rows, cols = image.shape

    if image is None:
        print('Image is None')

    for i in range(5):
            matrix = cv2.getRotationMatrix2D((rows/2, cols/2), rot_val, 1)
    new_img = cv2.warpAffine(image, matrix, (cols,rows))

    cv2.imshow('output', new_img)
    rotated_img = cv2.imwrite('rotated_img.jpg', new_img)

    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    return rotated_img
