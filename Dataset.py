import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def GetImages(folderpath, img_size):
    imgs = []

    # convert to grayscale since it's an mri scan
    for filename in sorted(os.listdir(folderpath)):
        img = load_img(os.path.join(folderpath, filename), color_mode="grayscale", target_size=img_size)
        imgs.append(img_to_array(img))

    return np.array(imgs)

def MakeDataset(img_size):
    # grabs all images with tumor
    imgs = GetImages('Dataset2/images/', img_size)
    masks = GetImages('Dataset2/masks/', img_size)
    labels = []

    if imgs.size != masks.size:
        return -1

    # since the img array contains only images with tumors, we have to add negative images
    imgs_no = GetImages('Dataset2/non_tumor/', img_size)
    masks_no = np.zeros(shape=imgs_no.shape)

    imgs = np.append(imgs, imgs_no, axis=0)
    masks = np.append(masks, masks_no, axis=0)

    for i in range(masks.shape[0]):
        labels.append(masks[i].max() > 0)

    return imgs.reshape(-1, img_size[0], img_size[1], 1), masks.reshape(-1, img_size[0], img_size[1], 1), np.asarray(labels)
