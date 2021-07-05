import argparse
import torch
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from skimage.transform import resize
from scipy import ndimage
import math

from PyTorch.model import PTModel

parser = argparse.ArgumentParser(
    description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='./PyTorch/nyu.pth',
                    type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='./examples/119_image.png',
                    type=str, help='Input filename or folder.')
args = parser.parse_args()

minDepth = 10
maxDepth = 1000


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

output = np.clip(my_DepthNorm(predictions.numpy(),
                 maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
depth = output[0, 0, :, :]


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

        vertices = []

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
                vertices.append([x, y, z])

                f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

        for u in range(0, img.shape[1]):
            for v in range(0, img.shape[0]):
                f.write("vt " + str(u/img.shape[1]) +
                        " " + str(v/img.shape[0]) + "\n")

        faces = []

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
                faces.append([v1-1, v2-1, v3-1])
                f.write("f " + vete(v3, v3) + " " +
                        vete(v2, v2) + " " + vete(v4, v4) + "\n")
                faces.append([v3-1, v2-1, v4-1])

        return np.array(vertices), np.array(faces)


v, f = create_obj(depth)

# image
rgb = np.asarray(pil_image)

# show
fig, axes = plt.subplots(2, 2)

axes[0][0].imshow(rgb)
axes[0][0].set_title("RGB")

axes[0][1].imshow(depth)
axes[0][1].set_title("Deep Map")

axes[1][0].axis('off')
axes[1][1].axis('off')

ax = fig.add_subplot(2, 1, 2, projection='3d')

# C = np.array([1 for _ in range(0, len(f))])
# norm = plt.Normalize(C.min(), C.max())
# colors = plt.cm.viridis(norm(C))
# pc = art3d.Poly3DCollection(v[f], facecolors=colors)
# ax.add_collection(pc)

# ax.set_xlim([min(a[0] for a in v), max(a[0] for a in v)])
# ax.set_ylim([min(a[1] for a in v), max(a[1] for a in v)])
# ax.set_zlim([min(a[2] for a in v), max(a[2] for a in v)])
# ax.view_init(-30, 90)


ax.set_title("3D (open the stored model.obj)")

plt.show()
