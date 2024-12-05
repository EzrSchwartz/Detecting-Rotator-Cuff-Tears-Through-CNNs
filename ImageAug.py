import imgaug.augmenters as iaa
import tqdm 
from tqdm import tqdm


seq = iaa.Sequential([
    iaa.rotate(-180,180),
    iaa.gaussian_noise(severity(2))
    
])