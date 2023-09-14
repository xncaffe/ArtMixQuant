from common.artMixCommonFunc import *

def get_initial_list(onnx_model):
    retlist = []
    for idx, initial in enumerate(onnx_model.graph.initializer):
        retlist.append(initial.name)
    return retlist

def get_precision_by_output(netJson, outputs):
    for output in outputs:
        for idx, tensor_name in enumerate(netJson.keys()):
            if output == tensor_name:
                return netJson[tensor_name]['bit_width'] 

def get_connection_net(netJson, opt_model):
    initialists = get_initial_list(opt_model)
    connectDict = {}
    netInputs = opt_model.graph.input
    for netInput in netInputs:
        connectDict[netInput.name] = {}
        connectDict[netInput.name]["output_tensor_bw"] = get_precision_by_output(netJson, [netInput.name])
        connectDict[netInput.name]["operator_type"] = "INPUT"
        connectDict[netInput.name]["input_tensor_names"] = []
        connectDict[netInput.name]["output_tensor_names"] = [netInput.name]
    for idx, node in enumerate(opt_model.graph.node):
        connectDict[node.name] = {}
        connectDict[node.name]["output_tensor_bw"] = get_precision_by_output(netJson, node.output)
        connectDict[node.name]["operator_type"] = node.op_type
        connectDict[node.name]["input_tensor_names"] = []
        connectDict[node.name]["output_tensor_names"] = []
        inputs = node.input
        for input in inputs:
            if input in initialists:
                continue
            elif not input:
                logger.warning('There is an empty input node "{}" in the OPT onnx model!'.format(node.name))
                continue
            else:
                connectDict[node.name]["input_tensor_names"].append(input)
        outputs = node.output
        for output in outputs:
            connectDict[node.name]["output_tensor_names"].append(output)
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
        logger.error("Not find opt model!!")
        assert(0)
    return optModelPath, optModelName

def get_output_tensors(NetJson):
    tensor_names=[]
    for idx, nodename in enumerate(NetJson):
        node=NetJson[nodename]
        output_tensor_names=node["output_tensor_names"]
        exist_flag=False
        for output_tensor_name in output_tensor_names:
            for fidx, fnodename in enumerate(NetJson):
                fnode=NetJson[fnodename]
                finput_tensor_names=fnode["input_tensor_names"]
                if output_tensor_name in finput_tensor_names:
                    exist_flag=True
                    break
            if not exist_flag:
                tensor_names.append(output_tensor_name)
                exist_flag=False
    return tensor_names