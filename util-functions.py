# -*- coding: utf-8 -*-
#https://www.programcreek.com/python/example/89219/keras.preprocessing.image.array_to_img


def show_image_path(img_path):
    import matplotlib.pyplot as plt
    import matplotlib.image as im
   
    img=im.imread(img_path)
    imgplot = plt.imshow(img)
    
    
def show_image(img):
    import matplotlib.pyplot as plt
    import matplotlib.image as im
    imgplot = plt.imshow(img)

#overload keras_image_augmentation
def keras_image_augmentation_frompath(source_img_path, dest_img_path):
    from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
    img = load_img(source_img_path) 
    keras_image_augmentation(img,dest_img_path)
    
def keras_image_augmentation(img, dest_img_path):
    from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img

    datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest')
    
     
    x = img_to_array(img)  # creating a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # converting to a Numpy array with shape (1, 3, 150, 150)


    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=dest_img_path, save_prefix='aug', save_format='jpeg'):
        i += 1
        if i > 1:
            break  # otherwise the generator would loop indefinitely

    
def preprocess_image_scale(image_path, img_size=None):
    '''
    Preprocess the image scaling it so that its larger size is max_size.
    This function preserves aspect ratio.
    '''
    #img = load_img(image_path)
    if img_size:
        scale = float(img_size) / max(img.size)
        new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
        img = img.resize(new_size, resample=Image.BILINEAR)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def resize_image(img, target_h, target_w, keep_aspect_ratio=False):
    from keras.preprocessing.image import img_to_array, array_to_img
    from PIL import Image as PILImage
    import numpy as np

    """
    Resizes an image to match target dimensions
    :type item: np.ndarray
    :type target_h: int
    :type target_w: int
    :param item: 3d numpy array or PIL.Image
    :param target_h: height in pixels
    :param target_w: width in pixels
    :param keep_aspect_ratio: If False then image is rescaled to smallest dimension and then cropped
    :return: 3d numpy array
    """
#    img = array_to_img(item, scale=False)
    if keep_aspect_ratio:
        img.thumbnail((target_w, target_w), PILImage.ANTIALIAS)
        img_resized = img
    else:
        img_resized = img.resize((target_w, target_h), resample=PILImage.NEAREST)

    # convert output
    img_resized = img_to_array(img_resized)
    img_resized = img_resized.astype(dtype=np.uint8)

    return img_resized 


def rotate_image(img, angle = 90):
    from scipy.ndimage import rotate
    #from scipy.misc import imread, imshow

    return rotate(img, angle)


def save_image(img,file_path, file_name):
    import os
    import matplotlib
    file_path = os.path.join(file_path,file_name)

    matplotlib.image.imsave(os.path.join(file_path),img)
    
def prespective_transform(source_img_path):
    import cv2
    import numpy as np
    
    img = cv2.imread(source_img_path)
    rows, cols, ch = img.shape    
    pts1 = np.float32(
        [[cols*.25, rows*.95],
         [cols*.90, rows*.95],
         [cols*.10, 0],
         [cols,     0]]
    )
    pts2 = np.float32(
        [[cols*0.1, rows],
         [cols,     rows],
         [0,        0],
         [cols,     0]]
    )    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    cv2.imshow('test', dst)
#    cv2.imwrite('zen.jpg', dst)
#    cv2.waitKey()
