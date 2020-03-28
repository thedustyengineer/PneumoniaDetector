import numpy as np
import cv2

class Perturbations:

    @staticmethod
    def adjust_brightness(image, brightness):
        return

    @staticmethod
    def rotate_image(image, rot_val):
        '''
        Rotate an image.

        Parameters
        ----------
        image: OpenCV image
            An OpenCV image (i.e. read with imread)
        degree: float
            The number of degrees to rotate
        
        Returns
        -------
        OpenCV image
            The input image but with the specified rotation
        '''
        rows, cols = image.shape
        matrix = cv2.getRotationMatrix2D((rows/2, cols/2), rot_val, 1)
        new_img = cv2.warpAffine(image, matrix, (cols,rows))
        return new_img

    @staticmethod
    def add_sp_noise(image, ratio, amount):
        '''
        Add salt and pepper noise to an image.

        Parameters
        ----------
        image: OpenCV image
            An OpenCV image (i.e. read with imread)
        ratio: float
            The ratio of salt to pepper (i.e. num_salt/num_pepper)
        
        Returns
        -------
        OpenCV image
            The input image but with salt and pepper noise applied
        '''
        rows, cols = image.shape
        
        img_copy = np.copy(image)
        salt_count = np.ceil(amount * image.size * ratio)
        coords = [np.random.randint(0, i - 1, int(salt_count))
            for i in image.shape] # build the salt coordinates
        img_copy[coords] = 255 # set the salt

        pepper_count = np.ceil(amount* image.size * (1. - ratio)) # 
        coords = [np.random.randint(0, i - 1, int(pepper_count))
                for i in image.shape] # build the pepper coords
        img_copy[coords] = 0 # set the pepper

        return img_copy



    @staticmethod
    def add_gauss_noise(image, sigma):
        '''
        Adds gaussian noise to an image.

        Parameters
        ----------
        image: OpenCV image
            An OpenCV image (i.e. read with imread)
        sigma: float
            The standard deviation of the gaussian distribution
        
        Returns
        -------
        OpenCV image
            The input image but with gaussian noise applied
        '''
        row,col = image.shape
        noisy_img = np.random.normal(0,sigma,(row,col))
        noisy_img = noisy_img.reshape(row,col)
        noisy_out = image + noisy_img
        return noisy_out
