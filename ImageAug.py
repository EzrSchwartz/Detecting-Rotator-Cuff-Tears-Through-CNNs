import imgaug.augmenters as iaa
import tqdm 
from tqdm import tqdm

import os as os
from PIL import Image

def extractImages(rootDirectory):
    for root,dirs,files in os.walk(rootDirectory):
        for dir in dirs:
            print(dir)
            for image in os.listdir(os.path.join(rootDirectory,dir)):
                if image.endswith(".jpg") or image.endswith(".png"):
                    print(image)

                    continue
                

seq = iaa.Sequential([
    # iaa.Rotate(,180),
    # iaa.gaussian_noise(severity(2))
    
])

extractImages("D:\ShoulderTears\ShoulderTest")

