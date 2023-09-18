
import numpy as np
import copy
from mixQuantEngine.fastMixCalculate import *

def del_invalid_param(setJson):
    outSetJson=copy.deepcopy(setJson)
    for _idx, setOpName in enumerate(outSetJson.keys()):
        cpOutSetJsonOp = copy.deepcopy(outSetJson[setOpName])
        for ix, paramName in enumerate(cpOutSetJsonOp.keys()):
            if paramName == "bit_width":
                continue
            del outSetJson[setOpName][paramName]
    return outSetJson

def mixQuantJson2Array(netJson):
    precision_array=[]
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        precision_array.append(node["output_tensor_bw"])
    return precision_array

def mixQuantJson2SearchArray(BlkNetJson):
    precision_array=[]
    for _index, blkname in enumerate(BlkNetJson.keys()):
        blkInfos=BlkNetJson[blkname]
        _param=[0,0]
        extFlag=False
        for nodeName in blkInfos.keys():
            node = blkInfos[nodeName]
            if(node["output_tensor_bw"]==8):
                extFlag=True
                break 
        if extFlag:
            _param[0] = 1
        else:
            _param[1] = 1               
        precision_array.append(_param)
    return precision_array

def Array2mixQuantJson(setJson, BlkInfos, precision_array, onnx_model, initialLists):
    outSetJson=del_invalid_param(setJson)
    setOpNameLists = []
    for _id, blkName in enumerate(BlkInfos.keys()):
        setBw = precision_array[_id]
        BlkNodes=BlkInfos[blkName]
        for _index, op_name in enumerate(BlkNodes.keys()):
            Node = BlkNodes[op_name]
            insertOpName = op_name + '_{}'.format(Node["operator_type"])
            if insertOpName in outSetJson.keys():
                if insertOpName not in setOpNameLists:
                    outSetJson[insertOpName]["bit_width"]=setBw
                    setOpNameLists.append(insertOpName)
            output_tensor_names = Node["output_tensor_names"]
            for output_tensor_name in output_tensor_names:
                setInitialLists = find_initializer_from_node(output_tensor_name, onnx_model, initialLists)
                for setInitialName in setInitialLists:
                    if setInitialName not in setOpNameLists:
                        outSetJson[setInitialName]["bit_width"]=setBw                      
                        setOpNameLists.append(setInitialName)
    return outSetJson

def mixQuantJson2Matrix(netJson):
    net_graph_code=[]
    one_hot_postion_type=["INPUT","CONV","POOL","ELTWISE_MUL","ELTWISE_ADD","CONCAT","SCALE","SOFTMAX","SIGMOID","MISH","TANH","CALLBACK"]
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        node_type=node["operator_type"]
        type_pos=one_hot_postion_type.index(node_type)
        current_one_hot_code=np.zeros(len(one_hot_postion_type)).tolist()
        current_one_hot_code[type_pos]=1
        net_graph_code.append(current_one_hot_code)
    return net_graph_code

def mixQuantJson2MatrixWithExtention(netJson):
    net_graph_code=[]
    one_hot_postion_type=["INPUT","CONV","POOL","ELTWISE_MUL","ELTWISE_ADD","CONCAT","SCALE","SOFTMAX","SIGMOID","MISH","TANH","CALLBACK"]
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        node_type=node["operator_type"]
        type_pos=one_hot_postion_type.index(node_type)
        current_one_hot_code=np.zeros(len(one_hot_postion_type)).tolist()
        current_one_hot_code[type_pos]=1
        if node_type=="INPUT":
            current_one_hot_code.extend([0,0])
        else:
            current_one_hot_code.extend([0,0])
        
        net_graph_code.append(current_one_hot_code)
    return net_graph_code