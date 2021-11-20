import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import ToTensor

import numpy as np
from PIL import Image
import math
from obj2html import obj2html

minDepth=10
maxDepth=1000
def my_DepthNorm(x, maxDepth):
    return maxDepth / x

def vete(v, vt):
    if v == vt:
        return str(v)
    return str(v)+"/"+str(vt)

def create_obj(img, objPath='model.obj', mtlPath='model.mtl', matName='colored', useMaterial=False):
    w = img.shape[1]
    h = img.shape[0]

    FOV = math.pi/4
    D = (img.shape[0]/2)/math.tan(FOV/2)

    if max(objPath.find('\\'), objPath.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)

    with open(objPath, "w") as f:
        if useMaterial:
            f.write("mtllib " + mtlPath + "\n")
            f.write("usemtl " + matName + "\n")

        ids = np.zeros((img.shape[1], img.shape[0]), int)
        vid = 1

        all_x = []
        all_y = []
        all_z = []

        for u in range(0, w):
            for v in range(h-1, -1, -1):

                d = img[v, u]

                ids[u, v] = vid
                if d == 0.0:
                    ids[u, v] = 0
                vid += 1

                x = u - w/2
                y = v - h/2
                z = -D

                norm = 1 / math.sqrt(x*x + y*y + z*z)

                t = d/(z*norm)

                x = -t*x*norm
                y = t*y*norm
                z = -t*z*norm

                f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

        for u in range(0, img.shape[1]):
            for v in range(0, img.shape[0]):
                f.write("vt " + str(u/img.shape[1]) +
                        " " + str(v/img.shape[0]) + "\n")

        for u in range(0, img.shape[1]-1):
            for v in range(0, img.shape[0]-1):

                v1 = ids[u, v]
                v3 = ids[u+1, v]
                v2 = ids[u, v+1]
                v4 = ids[u+1, v+1]

                if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                    continue

                f.write("f " + vete(v1, v1) + " " +
                        vete(v2, v2) + " " + vete(v3, v3) + "\n")
                f.write("f " + vete(v3, v3) + " " +
                        vete(v2, v2) + " " + vete(v4, v4) + "\n")

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.convA( torch.cat([up_x, concat_with], dim=1)  ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.densenet169( pretrained=False )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

model = PTModel().float()
path = "https://github.com/nicolalandro/DenseDepth/releases/download/0.1/nyu.pth"
model.load_state_dict(torch.hub.load_state_dict_from_url(path, progress=True))
model.eval()

def predict(inp):
    torch_image = ToTensor()(inp)
    images = torch_image.unsqueeze(0)

    with torch.no_grad():
       predictions = model(images)
    output = np.clip(my_DepthNorm(predictions.numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
    depth = output[0,0,:,:]

    img = Image.fromarray(np.uint8(depth*255))

    create_obj(depth, 'model.obj')
    html_string = obj2html('model.obj', html_elements_only=True)

    return img, html_string

st.title("Monocular Depth Estimation")

uploader = st.file_uploader('Upload your portrait here',type=['jpg','jpeg','png'])

if uploader is not None:
    pil_image = Image.open(uploader)
    pil_depth, html_string = predict(pil_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image)
    with col2:
        st.image(pil_depth)

    components.html(html_string)
    st.markdown(html_string, unsafe_allow_html=True)
