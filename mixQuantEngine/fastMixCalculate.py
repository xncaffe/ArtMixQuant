import os
import csv
import copy
from shutil import copyfile
# import multiprocessing 
from mixQuantEngine.fastEncoderDecoder import *
from mixQuantEngine.v2netEncoder import *
from mixQuantEngine.artRuntime import *
#from NormalProcessing import *

def find_initializer_from_node(nodeName, onnx_model, initialLists):
    correspondInitials = [nodeName]
    for idx, node in enumerate(onnx_model.graph.node):
        inputs = node.input
        if nodeName in inputs:
            for input in inputs:
                if input in initialLists:
                    correspondInitials.append(input)
    return correspondInitials

def set_bwjson_fast(setJson, BlkInfos, onnx_model, initialLists):
    outSetJson=copy.deepcopy(setJson)
    setOpNameLists = []
    for _id, blkName in enumerate(BlkInfos.keys()):
        BlkNodes=BlkInfos[blkName]
        for _index, op_name in enumerate(BlkNodes.keys()):
            Node = BlkNodes[op_name]
            insertOpName = op_name + '_{}'.format(Node["operator_type"])
            if insertOpName in outSetJson.keys():
                setOpNameLists.append(insertOpName)
            output_tensor_names = Node["output_tensor_names"]
            for output_tensor_name in output_tensor_names:
                setInitialLists = find_initializer_from_node(output_tensor_name, onnx_model, initialLists)
                for setInitialName in setInitialLists:
                    if setInitialName not in setOpNameLists:
                        setOpNameLists.append(setInitialName)
    for _idx, setOpName in enumerate(outSetJson.keys()):
        if setOpName in setOpNameLists:
            outSetJson[setOpName]["bit_width"] = 8
            cpOutSetJsonOp = copy.deepcopy(outSetJson[setOpName])
            for ix, paramName in enumerate(cpOutSetJsonOp.keys()):
                if paramName == "bit_width":
                    continue
                del outSetJson[setOpName][paramName]
        else:
            cpOutSetJsonOp = copy.deepcopy(outSetJson[setOpName])
            for ix, paramName in enumerate(cpOutSetJsonOp.keys()):
                if paramName != "bit_width":
                    del outSetJson[setOpName][paramName]
    return outSetJson

def get_mincycle_meetrmse(PerformDatas, targetRmse):
    print("############################################################################################")
    print("The fast method cannot achieve rmse and cycle to meet the set requirements at the same time!")
    print("Will match the lowest cycle hybrid method that first satisfies the target rmse!")
    print("############################################################################################")
    dstId=1
    PerformDatasList=list(PerformDatas.keys())
    for _index, blkIdName in enumerate(PerformDatasList):
        data=PerformDatas[blkIdName]
        rmse=data["sum_rmse"]
        if rmse>targetRmse:
            dstId=_index
            break
    dstBlkIdName=PerformDatasList[dstId-1]
    outData=PerformDatas[dstBlkIdName]
    return outData,dstId

def minquant_stack_eachblk(param, PerformPerBlk):
    targetRmse=param["targetRmse"]
    targetCycle=param["targetCycle"]
    output_base_dir=param["output_base_dir"]
    ini_inpath=param["ini_mix_path"]
    BlkNodeInfos=param["BlkNodeInfos"]
    artstudio_excutor_path=param["artstudio_excutor_path"]        
    mixQuantJsonHub=param["mixQuantJsonHub"]
    ini_validpath=ini_inpath[:-4]+"_fastprocessing.ini"
    setInitJson=param["setInitJson"]
    initialLists=param["initialLists"]
    onnx_model=param["opt_model"]
    dstFinalJsonDir=os.path.join(mixQuantJsonHub, get_ini_name(ini_inpath)+"_fast_rslt.json")
    setBlkInfos={}
    performance_datas={}
    endFlag=False
    dstId=0
    for _index, blkName in enumerate(PerformPerBlk.keys()):
        setBlkInfos[blkName]=BlkNodeInfos[blkName]
        if _index:
            newSetJson=set_bwjson_fast(setInitJson, setBlkInfos, onnx_model, initialLists)
            #newNetJson=precisionModify(newNetJson)
            dst_bwjson=os.path.join(mixQuantJsonHub, get_ini_name(ini_inpath)+"_stack_block_{}.json".format(_index))
            save_json(newSetJson, dst_bwjson)
            set_ini_bw_json(ini_inpath, ini_validpath, dst_bwjson)
            #run_artstudio(artstudio_excutor_path, ini_validpath, output_base_dir, param["UseSrc"], param["maskDump"])
            run_artstudio(artstudio_excutor_path, ini_validpath, output_base_dir, param["UseSrc"])
            output_dir=os.path.join(output_base_dir,get_ini_name(ini_validpath))
            _cycle,_rmse=get_total_cycle_mse(output_dir, param["netOutTensors"])
            _cossim=get_average_cossim(output_dir, param["netOutTensors"])
            data={"sum_cycle":_cycle, "sum_rmse":_rmse, "ave_cossim": _cossim}
            performance_datas[_index]=data
            if _cycle <= targetCycle and _rmse<=targetRmse:
                dstId=_index
                save_json(newSetJson, dstFinalJsonDir)
                endFlag=True
                #break
    if not endFlag:
        outData,dstId=get_mincycle_meetrmse(performance_datas, targetRmse)
        src_bwjson=os.path.join(mixQuantJsonHub, get_ini_name(ini_inpath)+"_stack_block_{}.json".format(dstId))
        try:
            copyfile(src_bwjson, dstFinalJsonDir)
        except:
            print("WRN: No suitable configuration was found. There may be a problem with ArtStudio. Please inform the developer to check!")
    else:
        outData=performance_datas[dstId]
    return outData, dstId, performance_datas

def loop_setprecision_perblk(param):
    performance_datas={}
    output_base_dir=param["output_base_dir"]
    ini_inpath=param["ini_mix_path"]
    BlkNodeInfos=param["BlkNodeInfos"]
    #NetJson=param["NetJson"]
    artstudio_excutor_path=param["artstudio_excutor_path"]
    mixQuantJsonHub=param["mixQuantJsonHub"]
    ini_validpath=ini_inpath[:-4]+"_fastprocessing.ini"
    setInitJson=param["setInitJson"]
    initialLists=param["initialLists"]
    onnx_model=param["opt_model"]
    dst_bwjson=os.path.join(mixQuantJsonHub, get_ini_name(ini_inpath)+"_per_block.json")
    for _index, blkName in enumerate(BlkNodeInfos.keys()):
        BlkInfos={}
        BlkInfos[blkName]=BlkNodeInfos[blkName]
        NewSetJson=set_bwjson_fast(setInitJson, BlkInfos, onnx_model, initialLists)
        #NewNetJson=precisionModify(NewNetJson)
        save_json(NewSetJson, dst_bwjson)
        set_ini_bw_json(ini_inpath, ini_validpath, dst_bwjson)
        #run_artstudio(artstudio_excutor_path, ini_validpath, output_base_dir, param["UseSrc"], param["maskDump"])
        run_artstudio(artstudio_excutor_path, ini_validpath, output_base_dir, param["UseSrc"], run_compiler=False)
        output_dir=os.path.join(output_base_dir,get_ini_name(ini_validpath))
        _cycle,_rmse=get_total_cycle_mse(output_dir, param["netOutTensors"], dump_cycle=False)
        _cossim=get_average_cossim(output_dir, param["netOutTensors"])
        data={"sum_cycle":_cycle, "sum_rmse":_rmse, "ave_cossim": _cossim}
        performance_datas[blkName]=data
    return performance_datas

def find_minrmse(inPerformPerBlk):
    minRmse=100000.0
    blkNameOfMinRmse=""
    for _index,BlkName in enumerate(inPerformPerBlk.keys()):
        performData=inPerformPerBlk[BlkName]
        if performData["sum_rmse"]<minRmse:
            minRmse=performData["sum_rmse"]
            blkNameOfMinRmse=BlkName
    return blkNameOfMinRmse

def qsort_rmse(inPerformPerBlk):
    outPerformPerBlk={}
    inPerformPerBlkCp=copy.deepcopy(inPerformPerBlk)
    for _index, BlkName in enumerate(inPerformPerBlk.keys()):
        minBlkName=find_minrmse(inPerformPerBlkCp)
        outPerformPerBlk[minBlkName]=inPerformPerBlk[minBlkName]
        if minBlkName in inPerformPerBlkCp:
            del inPerformPerBlkCp[minBlkName]
    return outPerformPerBlk    


