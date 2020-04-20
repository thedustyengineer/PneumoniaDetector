from pdetect_perturbations import Perturbations as pb
import cv2
import os
import numpy as np


def augment_data_set(input_dir, output_dir):
    '''
    Augments the data set using the perturbations in pdetect_perturbations.

    Parameters
    ----------
    input_dir: string
    The input directory containing only image files to be perturbed

    output_dir: string
    The output directory

    Returns
    -------
    No return value
    '''

    # initialize constants
    max_dim = 256
    perturbations = ['resize']

    file_num = 0 # initialize current file number
    file_count = len([name for name in os.listdir(input_dir) if (os.path.isfile(input_dir + '/' + name) and ('.jpeg' in name or '.jpg' in name))])
    for file in os.listdir(input_dir):

        if (('.jpeg' not in file) or ('.jpg' not in file) and ('._' in file)):
            continue


        current_percent = (file_num/file_count) * 100

        # load the file
        image = cv2.imread(input_dir + '/' + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # noise - must go first because it will have a much larger
        # impact after resizing
        if 'noise' in perturbations:
            image = pb.add_sp_noise(image, 0.50, 0.00001)
            image = pb.add_gauss_noise(image, 0.5)

        # apply perturbations
        if 'resize' in perturbations:
            image = pb.resize_image(image, 256)
            image = pb.pad_image(image, 256, 256)
            cv2.imwrite(output_dir + '/' + 'resized_' + file, image)
            
        # apply brightness perturbation
        if 'brightness' in perturbations:
            bness = np.random.uniform(-10, 10)
            bright_image = pb.adjust_brightness(image, 10)
            cv2.imwrite(output_dir + '/' + 'bright_image_' + file, bright_image)


        # save the non-flipped image and then save the flipped copy
        # if enabled
        cv2.imwrite(file + '.jpg', image)
        if 'flip' in perturbations:
            image = pb.mirror_image(image)
            cv2.imwrite(output_dir + '/' + 'flipped_' + file, image)
        
        # save the non-rotated image and then save the rotated copy
        # if enabled
        cv2.imwrite(file, image)

        if 'rotate' in perturbations:
            angle = np.random.uniform(-5, 5)
            image = pb.rotate_image(image, angle)
            cv2.imwrite(output_dir + '/' + 'rotated_' + file, image)
        
        image = None # clear out the image
        print('Finished file, currently ' + str(current_percent) + ' done.')
        file_num+=1



def build_save_np_binary(input_dir, output_dir, output_prefix):
    '''
    Builds and saves a numpy binary file for the specified directory.

    Parameters
    ----------
    input_dir: string
        The input directory containing only image files to be used for training/testing/validation

    output_dir: string
        The output directory for the np binary files

    output_prefix: string
        The prefix for the data set binaries (i.e. train, test, val)

    Returns
    -------
    x_train, y_train: tuple
        A tuple of the binary files for data and labels
    '''
    # initializing the numpy arrays that we will use to build the binary files
    # x_train is the image pixel data
    # y_train is the label data
    x_train = []
    y_train = []
    file_num = 0 # initialize current file number
    file_count = len([name for name in os.listdir(input_dir) if (os.path.isfile(input_dir + '/' + name) and ('.jpeg' in name or '.jpg' in name))])
    for file in os.listdir(input_dir):

        if (('.jpeg' not in file) or ('.jpg' not in file) and ('._' in file)):
            continue

        current_percent = (file_num/file_count) * 100

        # load the file
        image = cv2.imread(input_dir + '/' + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if (('normal' in file) or ('NORMAL' in file) or (('bacteria' not in file) and ('virus' not in file))):
<<<<<<< HEAD
            y_train.append(0)
        else:
            y_train.append(1)
=======
            y_train.append('normal')
        else:
            y_train.append('pneumonia')
>>>>>>> 35349a2aa2497bf6b581a2fb4c0823f3f2bb8bdf

        x_train.append(image)

        image = None # clear out the image
        print('Finished file, currently ' + str(current_percent) + ' done.')
        file_num+=1

    np.save(output_dir + '/' + 'x_' + output_prefix + '.npy', np.asarray(x_train))
    np.save(output_dir + '/' + 'y_' + output_prefix + '.npy', np.asarray(y_train))
            
    return x_train, y_train
    


#augment_data_set('/Volumes/Storage/chest_xray/chest_xray/test','/Volumes/Storage/chest_xray/chest_xray/test_resized')

<<<<<<< HEAD
build_save_np_binary('/Volumes/Storage/chest_xray/chest_xray/test_resized','/Volumes/Storage/chest_xray/chest_xray/test_resized/binaries','test')
=======
#build_save_np_binary('/Volumes/Storage/chest_xray/chest_xray/test_resized','/Volumes/Storage/chest_xray/chest_xray/test_resized/binaries','test')
>>>>>>> 35349a2aa2497bf6b581a2fb4c0823f3f2bb8bdf
