import OpenEXR
import numpy as np
import Imath
import array
from matplotlib import pyplot as plt

path = '/media/mint/Barracuda/Photogrammetry/meshroom/MeshroomCache/DepthMap/5a06ca41e99e8096ad3901fda8d4eef59c9a6f9c/30774549_depthMap.exr'

exr_opened = OpenEXR.InputFile(path)

for k, v in exr_opened.header().items():
    print(k, ':', v)
# Compute the size
dw = exr_opened.header()['dataWindow']
sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
data = array.array('f', exr_opened.channel('Y', FLOAT)).tolist()
depth = np.array(data).reshape(sz[1], sz[0])

plt.imshow(depth)
plt.show()

# SAVE SAMPLE
# npImage = np.squeeze(pilImage)
# size = img.shape
# exrHeader = OpenEXR.Header(size[1], size[0])

# exrHeader['channels'] = {"GRAY":Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT), 1, 1)}

# exrOut = OpenEXR.OutputFile("path/to/new.exr", exrHeader)
# GRAY = (npImage[:,:]).astype(np.float32).tobytes()
# exrOut.writePixels({'GRAY' : R})
# exrOut.close()
