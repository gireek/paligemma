from PIL import Image
import numpy as np
from typing import List

mean = [0.5]*3
std = [0.5]*3


# 1. function to resize the image to a specific size so it can be passed 
#    to the vision encoder
def resize(image, size, resample, reducing_gap):
    height, width = size
    resized_image = image.resize((width, height), resample, reducing_gap)
    return resized_image

# 2. function to rescale the image to be between 0 and 1 instead of 0 and 256
def rescale(image, rescale_factor):
    rescaled_image = image * rescale_factor
    return rescaled_image

# 3. function to standardize the image
def normalize(image, mean, std):
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    normalized_image = (image - mean)/std
    return normalized_image

# COMBINE IT ALL
def process_images(images: List[Image.Image], size, resample, reducing_gap, rescale_factor, mean, std):

    images = [resize(image, size, resample, reducing_gap) for image in images]

    images = [np.array(image) for image in images]

    images = [rescale(image, rescale_factor) for image in images]

    images = [normalize(image, mean, std) for image in images]

    # transpose here

    return images








# Paligemma isnt a normal VLM
# it can even do image segmentation and object detection
# how does that happen? special tokens!

