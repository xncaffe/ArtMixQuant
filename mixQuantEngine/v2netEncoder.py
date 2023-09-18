import os
import numpy as np
import json

def parse_json(jsonFile):
    fp = open(jsonFile,"r", encoding='utf-8')
    js_infos = json.load(fp)
    fp.close()
    return js_infos

def get_initialist(onnx_model):
    retlist = []
    for idx, initial in enumerate(onnx_model.graph.initializer):
        retlist.append(initial.name)
    return retlist

def get_precisionByOutput(netJson, outputs):
    for output in outputs:
        for idx, tensor_name in enumerate(netJson.keys()):
            if output == tensor_name:
                return netJson[tensor_name]['bit_width']    

def get_connection_net(netJson, opt_model):
    # opt_model=onnx.load(model_dir)
    initialists = get_initialist(opt_model)
    connectDict = {}
    netInputs = opt_model.graph.input
    for netInput in netInputs:
        connectDict[netInput.name] = {}
        connectDict[netInput.name]["output_tensor_bw"] = get_precisionByOutput(netJson, [netInput.name])
        connectDict[netInput.name]["operator_type"] = "INPUT"
        connectDict[netInput.name]["input_tensor_names"] = []
        connectDict[netInput.name]["output_tensor_names"] = [netInput.name]
    for idx, node in enumerate(opt_model.graph.node):
        #connectNodes = {}
        connectDict[node.name] = {}
        connectDict[node.name]["output_tensor_bw"] = get_precisionByOutput(netJson, node.output)
        # if "Art" not in node.op_type:
        #     connectDict[node.name]["operator_type"] = "CallBack"
        # else:
        connectDict[node.name]["operator_type"] = node.op_type
        connectDict[node.name]["input_tensor_names"] = []
        connectDict[node.name]["output_tensor_names"] = []
        inputs = node.input
        for input in inputs:
            if input in initialists:
                continue
            elif not input:
                print('MIXQUANTTOOL WRN: There is an empty input node "{}" in the OPT onnx model!'.format(node.name))
                continue
            else:
                # if not input:
                #     print('MIXQUANTTOOL ERROR: There is an empty input node "{}" in the OPT onnx model!'.format(node.name))
                #     assert(0)
                connectDict[node.name]["input_tensor_names"].append(input)
        outputs = node.output
        for output in outputs:
            connectDict[node.name]["output_tensor_names"].append(output)
        #connectJson.append(connectNodes)
    return connectDict, initialists  

def get_optmodel_path(outputdir):
    dirlists = os.listdir(outputdir)
    optModelPath=""
    optModelName=""
    for dirname in dirlists:
        if "-opt.onnx" in dirname:
            optModelPath=os.path.join(outputdir, dirname)
            optModelName=dirname
            break
    if not optModelName:
        print("ERR: Not find opt model!!")
        assert(0)
    return optModelPath, optModelName