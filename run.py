# -*- coding: utf-8 -*-

import os
from keras.preprocessing.image import load_img
from PIL import Image

source_img_path = "images/full_cabinet_view/"
dest_img_path = "images/full_cabinet_view/augmented/"

# Read all images from gear_images directory

for file in sorted(os.listdir(source_img_path)):
    if file.endswith(".jpg"):
        print('\t', file)
       
        file_path = os.path.join(source_img_path,file)
        img = load_img(file_path)  
        
        img = resize_image(img,4000, 4000, True)
        img = rotate_image(img, -90)
        
        show_image(img)
        save_image (img, dest_img_path,"norm_" + file )
        
        keras_image_augmentation(img, dest_img_path)
        #prespective_transform(source_img_path)