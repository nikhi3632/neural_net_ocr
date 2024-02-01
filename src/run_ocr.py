import os
import numpy as np
import pickle
import string
import skimage
from neural_net import forward, sigmoid, softmax
from optical_character_recognition import find_letters

ARTIFACTS_DIR = os.getcwd() + "/artifacts"
IMAGE_DIR = '../data/images'
PASS_IMAGES = [
    img_filename for img_filename in os.listdir(IMAGE_DIR)
    if any(ext in img_filename for ext in ['.png', '.jpg', '.jpeg']) and 'fail' not in img_filename
]

for img_filename in PASS_IMAGES:
    img_path = os.path.join(IMAGE_DIR, img_filename)
    # find the rows using..RANSAC, counting, clustering, etc.
    
    # crop the bounding boxes
    # note.. before flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get the images looking more like the dataset
    
    # load the weights
    # run the crops through the neural network and print the outputs
    
    # letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    # params = pickle.load(open('model_weights.pickle','rb'))
    
