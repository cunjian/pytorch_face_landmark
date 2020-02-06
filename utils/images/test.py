from PIL import Image
import numpy as np
img = Image.open('cifar.png')
pic = np.array(img)
noise = np.random.randint(-10,10,pic.shape[-1])
print(noise.shape)
pic = pic+noise
pic = pic.astype(np.uint8)
asd = Image.fromarray(pic)