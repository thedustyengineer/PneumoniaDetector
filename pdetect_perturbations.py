import numpy as np
import cv2

class Perturbations:

    @staticmethod
    def resize_image(image, max_1d_res):
        '''
        Resize an image proportionally based on a 
        new maximum res value for the larger dimension

        Parameters
        ----------
        image: OpenCV image
            An OpenCV image (i.e. read with imread)

        mrax_1d_res: int
            The maximum resolution of any one dimension
            (the smaller dimension will scale proportionally)
            
        Returns
        -------
        OpenCV image
            The resized input image
        '''
        x = None
        y = None

        if image.shape[0] > image.shape[1]: # if y > x
            y = max_1d_res
        else:
            x = max_1d_res

        width = x or int(image.shape[1] * (y/image.shape[0]))
        height = y or int(image.shape[0] * (x/image.shape[1]))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized

    @staticmethod
    def pad_image(image, x_pix, y_pix):
        '''
        Pad an image.

        Parameters
        ----------
        image: OpenCV image
            An OpenCV image (i.e. read with imread)
        x_pix: int
            The desired x-dimension in pixels
        y_pix: int
            The desired y-dimension in pixels
        Returns
        -------
        OpenCV image
            The padded input image
        '''
        padded_img = cv2.copyMakeBorder(image, x_pix, x_pix, y_pix, y_pix, cv2.BORDER_CONSTANT, value=0)
        return padded_img
    
    @staticmethod
    def adjust_brightness(image, brightness):
        '''
        Adjust brightness of an image.

        Parameters
        ----------
        image: OpenCV image
            An OpenCV image (i.e. read with imread)
            brightness: constant int
            The number of pixel changes
        
        Returns
        -------
        OpenCV image
            The input image with more accent on brightness depending on the pixel value added
        '''
        brightned_img = np.where((255 - image) < brightness, 255, image + brightness)
        brightned_img = image + brightness
        return brightned_img

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

    @staticmethod
    def mirror_image(image, axis=1):
        '''
        Mirrors an image over the specified axis 
        (0 - vertical, 1 - horizontal)

        Parameters
        ----------
        image: OpenCV image
            An OpenCV image (i.e. read with imread)
        
        Returns
        -------
        OpenCV image
            The input image but with gaussian noise applied
        '''

        flipped_image = cv2.flip(image,axis)
        return flipped_image