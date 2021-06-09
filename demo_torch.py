import argparse
import torch
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize
from scipy import ndimage
import math

from PyTorch.model import PTModel

parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='./PyTorch/nyu.pth', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='./examples/119_image.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

minDepth=10
maxDepth=1000
def my_DepthNorm(x, maxDepth):
    return maxDepth / x

model = PTModel().float()
model.load_state_dict(torch.load(args.model))
model.eval()

pil_image = Image.open(args.input)
torch_image = ToTensor()(pil_image)
images = torch_image.unsqueeze(0)

with torch.no_grad():
    predictions = model(images)

output = np.clip(my_DepthNorm(predictions.numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
depth = output[0,0,:,:]

def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)

def worldCoords(width, height):
    hfov_degrees, vfov_degrees = 57, 43
    hFov = math.radians(hfov_degrees)
    vFov = math.radians(vfov_degrees)
    cx, cy = width/2, height/2
    fx = width/(2*math.tan(hFov/2))
    fy = height/(2*math.tan(vFov/2))
    xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
    xx = (xx-cx)/fx
    yy = (yy-cy)/fy
    return xx, yy

def posFromDepth(depth, rgb):
    rgb_width, rgb_height = rgb.shape[1], rgb.shape[0]
    xx, yy = worldCoords(width=rgb_width//2, height=rgb_height//2)
    length = depth.shape[0] * depth.shape[1]

    depth[edges(depth) > 0.3] = 1e6  # Hide depth edges       
    z = depth.reshape(length)

    return np.dstack((xx*z, yy*z, z)).reshape((length, 3))

#image
rgb = np.asarray(pil_image)
# RGBD dimensions
width, height = depth.shape[1], depth.shape[0]
# Reshape
points = posFromDepth(depth.copy(), rgb.copy())
colors = resize(rgb, (height, width)).reshape((height * width, 3))
# Flatten and convert to float32
pos = points.astype('float32')
col = colors.reshape(height * width, 3).astype('float32') * 255

# show
fig, axes = plt.subplots(2,2)

axes[0][0].imshow(rgb)
axes[0][0].set_title("RGB")

axes[0][1].imshow(depth)
axes[0][1].set_title("Deep Map")

axes[1][0].axis('off')
axes[1][1].axis('off')

ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.set_title("Point Cloud")

plt.show()


