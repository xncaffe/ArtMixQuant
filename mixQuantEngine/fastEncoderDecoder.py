
import numpy as np
import os
import json
import copy

#INPUT_OUTPUT_PRECISION_SAME_NODES=["POOL","ELTWISE_MUL","SOFTMAX","SIGMOID","MISH","TANH","SOFTMAX_EXP","SOFTMAX_DIV", "CALLBACK"]
#INPUT_OUTPUT_PRECISION_SAME_NODES=["POOL","SOFTMAX","SOFTMAX_EXP","SOFTMAX_DIV", "CALLBACK"]
INPUT_OUTPUT_PRECISION_SAME_NODES=["POOL", "SOFTMAX_EXP", "CALLBACK"]
#FORCE_16BIT_NODES=["ELTWISE_MUL","SOFTMAX","SOFTMAX_EXP","SOFTMAX_DIV","SIGMOID","MISH","TANH"]

def parse_json(jsonFile):
    fp = open(jsonFile,"r", encoding='utf-8')
    js_infos = json.load(fp)
    fp.close()
    return js_infos

def get_net_output_nodes(NetJson):
    NetOutputNodes={}
    for node_index,op_name in enumerate(NetJson.keys()):
        _flag=False
        output_names=NetJson[op_name]["output_tensor_names"]
        for _index,name in enumerate(NetJson.keys()):
            input_names=NetJson[name]["input_tensor_names"]
            dupname_list=[x for x in output_names if x in input_names]
            if dupname_list:
                _flag=True
                break
        if not _flag:
            NetOutputNodes[op_name]=NetJson[op_name]
    return NetOutputNodes

def inter_duplicate_nodes(InputNodes):
    OutputNodes={}
    for _index, name in enumerate(InputNodes.keys()):
        if _index==0:
            OutputNodes[name]=InputNodes[name]
        else:
            dup_flag=False
            dup_names=[]
            for son_index, son_name in enumerate(OutputNodes.keys()):
                nameListIn=list(OutputNodes[son_name].keys())
                nameListOut=list(InputNodes[name].keys())
                dupname_list=[x for x in nameListIn if x in nameListOut]
                if dupname_list:
                    dup_names.append(son_name)
                    dup_flag=True
            if dup_flag:
                OutputNodesCp=copy.deepcopy(OutputNodes)
                for i, dup_name in enumerate(dup_names):
                    if i==0:
                        continue
                    else:
                        OutputNodes[dup_names[0]].update(OutputNodesCp[dup_name])
                        del OutputNodes[dup_name]
                OutputNodes[dup_names[0]].update(InputNodes[name])
            else:
                OutputNodes[name]=InputNodes[name]
    return OutputNodes

def loop_inter_dupgroup(NowOutputNodes):
    while 1:
        inLen=len(NowOutputNodes.keys())
        NowOutputNodes=inter_duplicate_nodes(NowOutputNodes)
        outLen=len(NowOutputNodes.keys())
        if inLen==outLen:
            break
    return NowOutputNodes

def get_precision_sameofinout(NetJson):
    NeedNodes={}
    id=0
    for node_index,op_name in enumerate(NetJson.keys()):
        operator=NetJson[op_name]["operator_type"]
        if operator in INPUT_OUTPUT_PRECISION_SAME_NODES:
            FocusNodes={}
            FocusNodes[op_name]=NetJson[op_name]
            InputOpNames=NetJson[op_name]["input_tensor_names"]
            for InputOpName in InputOpNames:
                if InputOpName not in NetJson.keys():
                    InputOpName=find_opname_from_outnames(NetJson, InputOpName)
                FocusNodes[InputOpName]=NetJson[InputOpName] ##可能需要添加本算子
            NeedNodes["iVoSameBlk_%d" % id]=FocusNodes
            id+=1
    NeedNodes=loop_inter_dupgroup(NeedNodes)
    return NeedNodes

def find_opname_from_outnames(NetJson, OutName):
    re_name=""
    for _index, op_name in enumerate(NetJson.keys()):
        output_names=NetJson[op_name]["output_tensor_names"]
        if OutName in output_names:
            re_name=op_name
            break
    if re_name:
        return re_name
    else:
        print(OutName+"is no corresponding op_name!")
        assert(0)

def find_opname_from_inputnames(NetJson, InputName):
    re_name=[]
    for _index, op_name in enumerate(NetJson.keys()):
        input_names=NetJson[op_name]["input_tensor_names"]
        if InputName in input_names:
            re_name.append(op_name)
    if re_name:
        return re_name
    else:
        print(InputName+"is no corresponding op_name!")
        assert(0)

def integrate_block(NetJson,NetOutputNodes):
    BlkNodeInfos={}
    NetJsonCp=copy.deepcopy(NetJson)
    for LastOutName in NetOutputNodes.keys():
        del NetJsonCp[LastOutName]
    OutputNodes=copy.deepcopy(NetOutputNodes)
    blk_id=0
    while 1:
        NowOutputNodes={}
        for _index,OutName in enumerate(OutputNodes.keys()):
            NowOutNode={}
            OutNode=OutputNodes[OutName]
            InputOfOutputNames=OutNode["input_tensor_names"]
            InputOfOutputOperator=OutNode["operator_type"]
            if InputOfOutputOperator!="CONCAT":
                for InOfOutName in InputOfOutputNames:
                    opname=find_opname_from_outnames(NetJson, InOfOutName)
                    InOfOutNode=NetJson[opname]
                    if opname not in NowOutNode.keys():
                        NowOutNode[opname]=InOfOutNode
                if NowOutNode:
                    NowOutputNodes["OutBlk_%d" % _index]=NowOutNode
            else:
                for _id,InOfOutName in enumerate(InputOfOutputNames):
                    #InOfOutNode=NetJson[InOfOutName]
                    if InOfOutName not in NetJson:
                        InOfOutName=find_opname_from_outnames(NetJson, InOfOutName)
                    InOfOutNode=NetJson[InOfOutName]
                    if InOfOutName not in NowOutNode.keys():
                        NowOutNode[InOfOutName]=InOfOutNode
                    if NowOutNode:
                        NowOutputNodes["OutBlk_{}_{}".format(_index, _id)]=NowOutNode
                        NowOutNode={}
        NowOutputNodes=loop_inter_dupgroup(NowOutputNodes)
        OutputNodes={}
        for _index, NowOutName in enumerate(NowOutputNodes.keys()):
            SonNodes=NowOutputNodes[NowOutName]
            BlkNodeInfos["blk_%d" % blk_id]=SonNodes
            blk_id+=1
            for SonId, SonNodeName in enumerate(SonNodes.keys()):
                if SonNodeName in NetJsonCp:
                    del NetJsonCp[SonNodeName]
                OutputNodes[SonNodeName]=SonNodes[SonNodeName]
        BlkNodeInfos=loop_inter_dupgroup(BlkNodeInfos)
        if len(NetJsonCp)<1:
            break
        BlkNodeInfos=loop_inter_dupgroup(BlkNodeInfos)
    return BlkNodeInfos

def inter_norspec_blk(NorBlkInfos, SpecBlkInfos):
    BlkNodeInfos={}
    OutInitInfos={}
    for NorIndex, NorBlkName in enumerate(NorBlkInfos.keys()):
        NorBlkNodes=NorBlkInfos[NorBlkName]
        for SpecIndex, SpecBlkName in enumerate(SpecBlkInfos.keys()):
            SpecBlkNodes=SpecBlkInfos[SpecBlkName]
            NorBlkList=list(NorBlkNodes.keys())
            SpecBlkList=list(SpecBlkNodes.keys())
            DupNameList=[x for x in NorBlkList if x in SpecBlkList]
            if DupNameList:
                NorBlkNodes.update(SpecBlkNodes)
        OutInitInfos[NorBlkName]=NorBlkNodes
    OutInitInfos=loop_inter_dupgroup(OutInitInfos)
    for Id,blk_name in enumerate(OutInitInfos.keys()):
        BlkNodeInfos["blk_%d" % Id]=OutInitInfos[blk_name]
    return BlkNodeInfos

def delete_specnodes(NetJson, BlkNodeInfos):
    BlkNodeInfosFn={}
    id=0
    for Index, BlkName in enumerate(BlkNodeInfos.keys()):
        BlkNodes=BlkNodeInfos[BlkName]
        DelFlag=False
        for i,name in enumerate(BlkNodes.keys()):
            Node=BlkNodes[name]
            #operator=Node["operator_type"]
            #input_tensor_names = Node["input_tensor_names"]
            output_tensor_names = Node["output_tensor_names"]
            #if operator in FORCE_16BIT_NODES:
            for output_tensor_name in output_tensor_names:
                delopnames = find_opname_from_inputnames(NetJson, output_tensor_name)
                for delopname in delopnames:
                    delNode = NetJson[delopname]
                    if (delNode["operator_type"] == "SCALE") and (len(delNode["input_tensor_names"]) >= 2):
                        DelFlag = True
                        break
                if DelFlag:
                    break
            # if (operator == "SCALE") and (len(input_tensor_names) >= 2):
            #     DelFlag=True
            #     break
        if not DelFlag:
            BlkNodeInfosFn["blk_%d" % id]=BlkNodes
            id+=1
    return BlkNodeInfosFn

def get_block_node_infos(NetJson):
    NetOutputNodes=get_net_output_nodes(NetJson)
    NorBlkNodeInfos=integrate_block(NetJson, NetOutputNodes)
    SpecBlkNodeInfos=get_precision_sameofinout(NetJson)
    BlkNodeInfos=inter_norspec_blk(NorBlkNodeInfos, SpecBlkNodeInfos)
    BlkNodeInfosFn=delete_specnodes(NetJson, BlkNodeInfos)
    return BlkNodeInfosFn