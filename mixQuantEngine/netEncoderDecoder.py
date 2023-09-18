
import numpy as np

def mixQuantJson2Array(netJson):
    precision_array=[]
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        precision_array.append(node["output_tensor_bw"])
    return precision_array

def mixQuantJson2SearchArray(netJson):
    precision_array=[]
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        _param=[0,0]
        if(node["output_tensor_bw"]==8):
            _param[0]=1
        else:
            _param[1]=1
        precision_array.append(_param)
    return precision_array

def Array2mixQuantJson(netJson,precision_array):
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        node["output_tensor_bw"]=precision_array[node_index]
    return netJson

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