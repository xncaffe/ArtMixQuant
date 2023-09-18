import argparse
import os

from NormalProcessing import *
from FastProcessing import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", default="/workspace/nxu/ArtStudio/Art.Studio-dev-0830/", help="artstudio root path")
    parser.add_argument("-ini", default="/workspace/nxu/ArtStudio/Art.Studio-dev-0830/data/sample/sampleArtProject/model/resnet18_onnx.ini", help="artstudio ini path")
    parser.add_argument("-o", default="/workspace/nxu/ArtStudio/Art.Studio-dev-0830/data/sample/sampleArtProject/output", help="artstudio output dir")
    parser.add_argument("-w", default=10, help="multi process workers")
    parser.add_argument("-pm", default=1, help="mix precision processing method, 1=standard, 0=fast")
    parser.add_argument("-m", default=20, help="sample group, valid when pm=1")
    parser.add_argument("-s", default=0, help="test_slave, valid when pm=1")
    parser.add_argument("-r", default=10000, help="search round, valid when pm=1")
    parser.add_argument("-rd", default=0, help="random bais, set 16 or 8 or 0, valid when pm=1")
    parser.add_argument("-d", default=1, help="mix precision set direction, set 1 or 0, valid when pm=1")
    parser.add_argument("--PF", action='store_true', help="mse penalty factor or not, valid when pm=1")
    parser.add_argument("--UseOpt", action='store_true', help="use opt-model or no when mix quant search. Only support ONNX!")
    parser.add_argument("-xovr", default=0.1, help="xovr, valid when pm=1")
    parser.add_argument("-mp", default=0.85, help="mp, valid when pm=1")
    parser.add_argument("-cr", default=0.5, help="target cycle ratio, less than 1, valid when pm=0")
    parser.add_argument("-mr", default=0.8, help="target rmse ratio, less than 1, valid when pm=0")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    create_dir(args.o)
    method_p = int(args.pm)
    if method_p==1:
        normal_process(args)
    elif method_p==0:
        fast_process(args)
    else:
        print("The parameter 'pm' setting is invalid, please check!")

if __name__ == "__main__":
    main()
