import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.transforms import ToTensor

import numpy as np
from PIL import Image
import Imath
import OpenEXR
import argparse
from io import StringIO
import json
import os

minDepth = 10
maxDepth = 1000
model_url = 'https://github.com/nicolalandro/DenseDepth/releases/download/0.1/nyu.pth'

parser = argparse.ArgumentParser(description='DenseDepth for Meshroom.')
parser.add_argument('--input', help='input folder')
parser.add_argument('--preparedDenseScene', help='prepared exp folder')
parser.add_argument('--output', help='output folder')


args = parser.parse_args()

def main():
    model = PTModel().float()
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
    model.eval()
    # load to gpu if cuda

    # Read args
    with open(args.input, 'r') as json_file:
        input_json = json.load(json_file)

    # for each image in folder
    max_length = len(input_json['views'])
    for i, data in enumerate(input_json['views']):
        print(f'{i+1}/{max_length}')
        image_path = data['path']

        pil_image = Image.open(image_path)
        torch_image = ToTensor()(pil_image)
        images = torch_image.unsqueeze(0)

        # load to gpu if cuda

        with torch.no_grad():
            # if cuda use a batch of images to speed up the process
            predictions = model(images)

        output = np.clip(my_DepthNorm(predictions.numpy(),
                        maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
        depth = output[0, 0, :, :]

        # store .exr and .log file
        npImage = np.squeeze(depth)
        size = depth.shape
        exrHeader = OpenEXR.Header(size[1], size[0])

        exrHeader['channels'] = {
            "Y": Imath.Channel(
                Imath.PixelType(Imath.PixelType.FLOAT),
                1,
                1,
            )
        }
        exrHeader['AliceVision:downscale'] = 2
        exrHeader['AliceVision:maxDepth'] = maxDepth
        exrHeader['AliceVision:minDepth'] = minDepth

        output_path = os.path.join(args.output, data['viewId'] + '_depthMap.exr')
        exrOut = OpenEXR.OutputFile(output_path, exrHeader)
        Y = (npImage[:, :]).astype(np.float32).tobytes()
        exrOut.writePixels({'Y': Y})
        exrOut.close()


def my_DepthNorm(x, maxDepth):
    return maxDepth / x


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features,
                               kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(
            output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(
            2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))


class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features,
                               kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256,
                            output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,
                            output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,
                            output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,
                            output_features=features//16)

        self.conv3 = nn.Conv2d(
            features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[
            3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet169(pretrained=False)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    main()
