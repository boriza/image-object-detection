# -*- coding: utf-8 -*-
from keras.preprocessing.image import load_img
from PIL import Image

source_img_path = "images/full_cabinet_view/IMG_1070.jpg"
dest_img_path = "images/full_cabinet_view/augmented/"

#show_image(source_img_path)

filename = "IMG_1070.jpg"

img = load_img(source_img_path)  


img = resize_image(img,4000, 4000, True)
img = rotate_image(img, -90)

show_image(img)
save_image (img, dest_img_path,"norm_" + filename )

keras_image_augmentation(img, dest_img_path)
#prespective_transform(source_img_path)
