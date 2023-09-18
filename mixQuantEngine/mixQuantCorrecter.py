'''
Description: 
version: 1.0
Author: gong hao, Artosyn
Date: 2022-01-06
'''
import copy
import numpy as np
from mixQuantEngine.artRuntime import *

INPUT_OUTPUT_PRECISION_SAME_NODES=["POOL","ELTWISE_MUL","SOFTMAX","SIGMOID","MISH","TANH"]
FORCE_16BIT_NODES=["ELTWISE_MUL","SOFTMAX","SOFTMAX_EXP","SOFTMAX_DIV","SIGMOID","MISH","TANH"]

def updatePrecisionByOutput(netJson,tensor_name,bw):
    # update output tensors by reverse traverse
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        if(tensor_name in node["output_tensor_names"]):
            print(tensor_name,node["output_tensor_bw"],"----->",bw)
            node["output_tensor_bw"]=bw

##### Add by NanXu in 2022-05-20 #####
def updateConcatInternal(node,netJson,bw):
    if len(node["input_tensor_names"])<=1:
        node["internal_bw(read_only)"]=bw
    else:
        bw_flag = 1
        for input_tensor in node["input_tensor_names"]:
            input_node = netJson[input_tensor]
            if input_node["output_tensor_bw"] != bw:
                bw_flag = 0
        if bw_flag:
            node["internal_bw(read_only)"]=bw

def updateInternalByOutput(netJson, tensor_name, bw):
    keys=list(netJson.keys())
    keys.reverse()
    for node_index,name in enumerate(keys):
        node=netJson[name]
        if tensor_name in node["input_tensor_names"]:
            if node["internal_bw(read_only)"] != bw:
                if node["operator_type"] in "CONCAT":
                    updateConcatInternal(node,netJson,bw)
                else:
                    node["internal_bw(read_only)"] == bw

def updateInOutPrecisionSync(netJson):
    keys=list(netJson.keys())
    keys.reverse()
    for node_index,name in enumerate(keys):
        node=netJson[name]
        if(node["operator_type"] in INPUT_OUTPUT_PRECISION_SAME_NODES):
            # modify previous input operator tensors according to output tensor
            input_bw = node["internal_bw(read_only)"]
            output_bw = node["output_tensor_bw"]
            if(input_bw != output_bw):
                for input_tensor in node["input_tensor_names"]:
                    input_node = netJson[input_tensor]
                    input_node_output_bw = input_node["output_tensor_bw"]
                    if input_node_output_bw == input_bw:
                        print(name,node["output_tensor_bw"],"----->",input_bw)
                        node["output_tensor_bw"] = input_bw
                        updateInternalByOutput(netJson, name, node["output_tensor_bw"])
                    else:
                        node["internal_bw(read_only)"] = output_bw
            else:
                continue
    return netJson            
#######################################

def getPrecisionByOutput(netJson,tensor_name):
    # update output tensors by reverse traverse
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        if(tensor_name in node["output_tensor_names"]):
            return node["output_tensor_bw"]

##### Add by NanXu in 2022-06-10 #####
def updateCollectPrecision(CollectNodes,netJson):
    precisionBit=[0,0]
    bw=[8,16]
    ChoseBit=bw[0]
    for _index,CollectNodeName in enumerate(CollectNodes.keys()):
        precision=CollectNodes[CollectNodeName]["output_tensor_bw"]
        if precision==bw[0]:
            precisionBit[0]=precisionBit[0]+1
        else:
            precisionBit[1]=precisionBit[1]+1
    if precisionBit[0]==precisionBit[1]:
        ChoseBit=bw[np.random.choice([0,1])]
    elif precisionBit[0]>precisionBit[1]:
        ChoseBit=bw[0]
    else:
        ChoseBit=bw[1]
    for _index,CollectNodeName in enumerate(CollectNodes.keys()):
        updatePrecisionByOutput(netJson,CollectNodeName,ChoseBit)
        updateInternalByOutput(netJson, CollectNodeName,ChoseBit)

def updateDaulInputPrecision(netJson, nodeName):
    node=netJson[nodeName]
    mOutFlag=False
    unifNodes={}
    ConcatNodes=[nodeName]
    inNames=node["input_tensor_names"]
    idx=0
    while 1:
        initInNames=[]
        checkNodes={}
        for inName in inNames:
            if inName not in unifNodes:
                unifNodes[inName]=netJson[inName]
            for _index, name in enumerate(netJson.keys()):
                input_tensor_names=netJson[name]["input_tensor_names"]
                operator_type=netJson[name]["operator_type"]
                if operator_type=="CONCAT":
                    continue
                if inName in input_tensor_names:
                    if name not in ConcatNodes:
                        checkNodes[name]=netJson[name]
        for id, checkNodeName in enumerate(checkNodes.keys()):
            ConcatNodes.append(checkNodeName)
            input_tensor_names=netJson[checkNodeName]["input_tensor_names"]
            if len(input_tensor_names)>1:
                for input_tensor_name in input_tensor_names:
                    if input_tensor_name not in inNames:
                        initInNames.append(input_tensor_name)
        inNames=copy.deepcopy(initInNames)
        if mOutFlag:
            break
        if len(checkNodes)==len(initInNames):
            mOutFlag=True 
        idx+=1
        if idx>100:
            print("ERROR! The project has fallen into an infinite loop and has been forcibly exited!!")
            print("checkNodes: ", checkNodes)
            print("initInNames:", initInNames)       
    updateCollectPrecision(unifNodes,netJson)
#######################################

def updateTensorPrecision(netJson):
    keys=list(netJson.keys())
    keys.reverse()
    # step 1, force 16 bit add check input align by left branch, update output tensors by reverse traverse
    for node_index,name in enumerate(keys):
        node=netJson[name]
        if node["operator_type"]=="SLICE":
            print("ERROR! not support slice!")
            assert(0)
        if(node["operator_type"] in FORCE_16BIT_NODES):
            # modify previous input operator tensors
            for input_tensor in node["input_tensor_names"]:
                updatePrecisionByOutput(netJson,input_tensor,16)
            # force 16 bit output tensor
            node["output_tensor_bw"]=16
        # SCALE has two inputs, the second must be 16 bw
        elif node["operator_type"]=="SCALE":
            if len(node["input_tensor_names"])==2:
                weight_position_tensor=node["input_tensor_names"][1]
                updatePrecisionByOutput(netJson,weight_position_tensor,16)
        elif node["operator_type"] == "CALLBACK":
            # modify previous input operator tensors
            for input_tensor in node["input_tensor_names"]:
                updatePrecisionByOutput(netJson,input_tensor,node["output_tensor_bw"])
        # ELTWISE_ADD has same precision input by volting with output
        elif node["operator_type"]=="ELTWISE_ADD":
            left_bw=getPrecisionByOutput(netJson,node["input_tensor_names"][0])
            right_bw=getPrecisionByOutput(netJson,node["input_tensor_names"][1])
            if left_bw!=right_bw:
                ##### Repair by NanXu in 2022-06-10 #####
                updateDaulInputPrecision(netJson, name)
                #if left_bw==node["output_tensor_bw"]:
                #    updatePrecisionByOutput(netJson,node["input_tensor_names"][1],left_bw)
                #if right_bw==node["output_tensor_bw"]:
                #    updatePrecisionByOutput(netJson,node["input_tensor_names"][0],right_bw)
                ##########################################
        ##### Add by NanXu in 2022-05-17 #####
        elif node["operator_type"]=="CONV" and len(node["input_tensor_names"])>1:
            left_bw=getPrecisionByOutput(netJson,node["input_tensor_names"][0])
            right_bw=getPrecisionByOutput(netJson,node["input_tensor_names"][1])
            if left_bw!=right_bw:
                if left_bw!=node["internal_bw(read_only)"]:
                    updatePrecisionByOutput(netJson,node["input_tensor_names"][0],node["internal_bw(read_only)"])
                if right_bw!=node["internal_bw(read_only)"]:
                    updatePrecisionByOutput(netJson,node["input_tensor_names"][1],node["internal_bw(read_only)"])
        #######################################
    # step 2, align input output precision, update output tensors by reverse traverse
    for node_index,name in enumerate(keys):
        node=netJson[name]
        if(node["operator_type"] in INPUT_OUTPUT_PRECISION_SAME_NODES):
            # modify previous input operator tensors according to output tensor
            for input_tensor in node["input_tensor_names"]:
                updatePrecisionByOutput(netJson,input_tensor,node["output_tensor_bw"])
        
    return netJson

def updateOperatorPrecision(netJson):
    for node_index,name in enumerate(netJson.keys()):
        node=netJson[name]
        if(node["operator_type"] == "CONCAT" or node["operator_type"] == "CALLBACK"):
            bw_same_flag=True
            for input_tensor in node["input_tensor_names"]:
                if getPrecisionByOutput(netJson,input_tensor)!=node["output_tensor_bw"]:
                    bw_same_flag=False
            if bw_same_flag:
                node["internal_bw(read_only)"]=node["output_tensor_bw"]
        elif node["operator_type"]=="ELTWISE_ADD":
            first_input_bw=getPrecisionByOutput(netJson,node["input_tensor_names"][0])
            for input_tensor in node["input_tensor_names"][1:]:
                if first_input_bw!=getPrecisionByOutput(netJson,input_tensor):
                    print("ERROR! INPUT TENSORS ARE NOT SAME! NODE:",node)
                    assert(0)
            node["internal_bw(read_only)"]=getPrecisionByOutput(netJson,node["input_tensor_names"][0])
        elif node["operator_type"]=="INPUT":
            node["internal_bw(read_only)"]=node["output_tensor_bw"]
        else:
            first_input_bw=getPrecisionByOutput(netJson,node["input_tensor_names"][0])
            for input_tensor in node["input_tensor_names"][1:]:
                if first_input_bw!=getPrecisionByOutput(netJson,input_tensor):
                    print("ERROR! INPUT TENSORS ARE NOT SAME! NODE:",node)
                    assert(0)
            node["internal_bw(read_only)"]=getPrecisionByOutput(netJson,node["input_tensor_names"][0])
    return netJson

def precisionModify(netJson):
    # update tensor precision according to operator type limitation
    netJsonLast=copy.deepcopy(netJson)
    netJson=copy.deepcopy(updateTensorPrecision(netJson))
    while(netJsonLast!=netJson):
        netJsonLast=copy.deepcopy(netJson)
        netJson=copy.deepcopy(updateTensorPrecision(netJson))
    # update operator read_only precision 
    netJson=updateOperatorPrecision(netJson)
    ##### Add by NanXu in 2022-05-20 #####
    # update input sync output when the operator must be required. INPUT_OUTPUT_PRECISION_SAME_NODES 
    netJson=updateInOutPrecisionSync(netJson)
    #######################################
    return netJson

