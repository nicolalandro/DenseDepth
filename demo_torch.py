import argparse
import torch
import glob
import numpy as np
from matplotlib import pyplot as plt

from utils import load_images
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

inputs = load_images( glob.glob(args.input) ).astype('float32')
pytorch_input = torch.from_numpy(inputs[0,:,:,:]).permute(2,0,1).unsqueeze(0)
images = pytorch_input[0,:,:,:].unsqueeze(0)

with torch.no_grad():
    predictions = model(images)

output = np.clip(my_DepthNorm(predictions.numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

plt.imshow(output[0,0,:,:])
plt.show()
