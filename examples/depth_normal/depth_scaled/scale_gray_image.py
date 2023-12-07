import os
from PIL import Image
import glob
import numpy as np

files = glob.glob('examples/depth_normal/depth/*')

for file in files:
    img = Image.open(file)
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    # img = img / img.max() * 255
    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    img.save(file.replace('depth_normal/depth', 'depth_normal/depth_scaled'))