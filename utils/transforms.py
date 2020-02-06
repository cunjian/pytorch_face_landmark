from __future__ import division
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
class RandomRotate(object):
    def __init__(self,range):
        self.range = range
        assert len(range) == 2
    def __call__(self, img):
        angle = np.random.randint(self.range[0],self.range[1],1)
        return img.rotate(angle)

class RandomJitter(object):
    def __init__(self,range):
        self.range = range
        assert len(range) == 2
    def __call__(self, img):
        pic = np.array(img)
        noise = np.random.randint(self.range[0],self.range[1],pic.shape[-1])
        pic = pic+noise
        pic = pic.astype(np.uint8)
        return Image.fromarray(pic)






