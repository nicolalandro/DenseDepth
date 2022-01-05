[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolalandro/DenseDepth/blob/master/DenseDepth_torch.ipynb)
[![Open Streamlit in HuggingFace spaces](https://img.shields.io/badge/Streamlit-Open%20in%20Spaces-blueviolet)](https://huggingface.co/spaces/z-uo/monocular_depth_estimation)

# Monocular Depth Estimation
This is a fork of [DenseDepth](https://github.com/ialhashim/DenseDepth).


## Requirements
```
# python3.6 and cuda
sudo apt-get install python3-opencv
python3 -m pip install -r requirements.txt
```

## Exec
```
wget https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5
python3 demo.py
```

<p align="center">
  <img style="max-width:500px" src="https://s3-eu-west-1.amazonaws.com/densedepth/densedepth_results_04.jpg" alt="RGBD Demo">
</p>

## Failed ONNX test
```
python3 model2onnx.py
python3 check_onnx_model.py
```

## Torch test
* transform weight to torch (edit the file)
```
cd PyTorch
python3 load_weight_from_keras.py
```
* use torch only
```
# download or generate torch weight nyu.pth into the PyTorch folder
python3 demo_torch.py
```

