__version__ = "2.0"

from meshroom.core import desc


class DenseDepthDeepMap(desc.CommandLineNode):
    commandLine = 'python /media/mint/Barracuda/Models/DenseDepth/meshroom.py {allParams}'
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize('input')
    parallelization = desc.Parallelization(blockSize=3)
    commandLineRange = '--rangeStart {rangeStart} --rangeSize {rangeBlockSize}'

    category = 'DenseDepth '
    documentation = '''
For each camera that have been estimated by the Structure-From-Motion, it estimates the depth value per pixel.
Adjust the downscale factor to compute depth maps at a higher/lower resolution.
Use a downscale factor of one (full-resolution) only if the quality of the input images is really high (camera on a tripod with high-quality optics).
## Online
[https://alicevision.org/#photogrammetry/depth_maps_estimation](https://alicevision.org/#photogrammetry/depth_maps_estimation)
'''

    inputs = [        
        desc.File(
            name='imagesFolder',
            label='Images Folder',
            description='Use images from a specific folder instead of those specify in the SfMData file.\nFilename should be the image uid.',
            value='',
            uid=[0],
        ),
       
        desc.ChoiceParam(
            name='verboseLevel',
            label='Verbose Level',
            description='''verbosity level (fatal, error, warning, info, debug, trace).''',
            value='info',
            values=['fatal', 'error', 'warning', 'info', 'debug', 'trace'],
            exclusive=True,
            uid=[],
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output',
            description='Output folder for generated depth maps.',
            value=desc.Node.internalFolder,
            uid=[],
        ),
    ]