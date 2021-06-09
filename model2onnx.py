from demo import load_model
try:
    import keras2onnx
except:
    print('Please install keras2onnx for example with `pip install keras2onnx`')
    exit()


if __name__ == '__main__':
    print('Start...')
    
    model = load_model()
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras2onnx.save_model(onnx_model, 'nyu.onnx')
