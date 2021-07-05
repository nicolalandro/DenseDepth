__version__ = "3.0"

from meshroom.core import desc


class DenseDepthDenseMap(desc.CommandLineNode):
    commandLine = 'python3.6 /media/mint/Barracuda/Models/DenseDepth/meshroom/main.py {allParams}'

    cpu = desc.Level.NORMAL
    ram = desc.Level.NORMAL

    inputs = [
        desc.File(
            name="input",
            label='Images Folder',
            description='',
            value='',
            uid=[0],
            ),
        desc.File(
            name="preparedDenseScene",
            label='Prepared expr camera and other info',
            description='',
            value='',
            uid=[1],
            ),
]

    outputs = [
        desc.File(
            name="output",
            label="Output depthmap",
            description="Output folder for generated depth maps.",
            value=desc.Node.internalFolder,
            uid=[],
            ),
    ]

