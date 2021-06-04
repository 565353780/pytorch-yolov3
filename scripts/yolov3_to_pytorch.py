import numpy as np
import torch
from torch._C import PyTorchFileReader
# import torchsnooper
from models import Darknet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# with torch.jit.optimized_execution(True):

def save_pytorch_model(darknet_model,pytorch_model,config_file):

    model = Darknet(config_file).to(device)
    model.load_darknet_weights(darknet_model)
    torch.save(model.state_dict(), pytorch_model)

    print('pytorch model saved!')

def convert_pytorch_model_to_libtorch(config_file,pytorch_model,libtorch_model):

    model = Darknet(config_file).to(device)

    model.load_state_dict(torch.load(pytorch_model))

    model.eval()
    example = torch.rand(1,3,416,416).cuda()
    with torch.jit.optimized_execution(True):
        print('pp')
        # example 报错 Expected type 'tuple', got 'Tensor' instead ，可能是input没有放cuda上
        traced_script_module = torch.jit.trace(model, example,check_trace=False)# save the converted model
        print('oo')
        traced_script_module.save(libtorch_model)

        output = traced_script_module(torch.rand(1,3,416,416).cuda())
        print(output.shape)

    print('trace model saved!')
    
if __name__ == "__main__":
    config_file = "yolov3_train_waterDrop_2class/yolov3.cfg"
    darknet_model = "yolov3_train_waterDrop_2class/yolov3_train_waterDrop_2class.weights"
    
    pytorch_model = darknet_model.split(".")[0] + ".pth"
    libtorch_model = darknet_model.split(".")[0] + ".pt"
    
    save_pytorch_model(darknet_model, pytorch_model, config_file)
    
    convert_pytorch_model_to_libtorch(config_file, pytorch_model, libtorch_model)