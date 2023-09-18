import os
import csv
import copy
from shutil import copyfile
# import multiprocessing 
from mixQuantEngine.fastEncoderDecoder import *
from NormalProcessing import *

def set_bwjson_fast(NetJson, BlkInfos):
    outNetJson=copy.deepcopy(NetJson)
    for _id, blkName in enumerate(BlkInfos.keys()):
        BlkNodes=BlkInfos[blkName]
        for _index, op_name in enumerate(BlkNodes.keys()):
            outNetJson[op_name]["output_tensor_bw"]=8
            output_tensor_names=outNetJson[op_name]["output_tensor_names"]
            for _id, name in enumerate(outNetJson.keys()):
                input_tensor_names=outNetJson[name]["input_tensor_names"]
                dupname_list=[x for x in output_tensor_names if x in input_tensor_names]
                #if op_name in input_tensor_names:
                if dupname_list:
                    if outNetJson[name]["internal_bw(read_only)"] != 8:
                        outNetJson[name]["internal_bw(read_only)"]=8
    return outNetJson

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
    NetJson=param["NetJson"] 
    artstudio_excutor_path=param["artstudio_excutor_path"]        
    mixQuantJsonHub=param["mixQuantJsonHub"]
    ini_validpath=ini_inpath[:-4]+"_fastprocessing.ini"
    dstFinalJsonDir=os.path.join(mixQuantJsonHub, get_ini_name(ini_inpath)+"_fast_rslt.json")
    setBlkInfos={}
    performance_datas={}
    endFlag=False
    dstId=0
    for _index, blkName in enumerate(PerformPerBlk.keys()):
        setBlkInfos[blkName]=BlkNodeInfos[blkName]
        if _index:
            newNetJson=set_bwjson_fast(NetJson, setBlkInfos)
            newNetJson=precisionModify(newNetJson)
            dst_bwjson=os.path.join(mixQuantJsonHub, get_ini_name(ini_inpath)+"_stack_block_{}.json".format(_index))
            save_json(newNetJson, dst_bwjson)
            set_ini_bw_json(ini_inpath, ini_validpath, dst_bwjson)
            run_artstudio(artstudio_excutor_path, ini_validpath, output_base_dir)
            output_dir=os.path.join(output_base_dir,get_ini_name(ini_validpath))
            _cycle,_rmse=get_total_cycle_mse(output_dir)
            data={"sum_cycle":_cycle, "sum_rmse":_rmse}
            performance_datas[_index]=data
            if _cycle <= targetCycle and _rmse<=targetRmse:
                dstId=_index
                save_json(newNetJson, dstFinalJsonDir)
                print("############################################################################################")
                print("The mixed quantification is completed, and the target rmse and cycle have been reached!")
                print("Writing mix quant results..........................................")
                print("############################################################################################")
                endFlag=True
                break
    if not endFlag:
        outData,dstId=get_mincycle_meetrmse(performance_datas, targetRmse)
        src_bwjson=os.path.join(mixQuantJsonHub, get_ini_name(ini_inpath)+"_stack_block_{}.json".format(dstId))
        copyfile(src_bwjson, dstFinalJsonDir)
    else:
        outData=performance_datas[dstId]
    return outData, dstId

def loop_setprecision_perblk(param):
    performance_datas={}
    output_base_dir=param["output_base_dir"]
    ini_inpath=param["ini_mix_path"]
    BlkNodeInfos=param["BlkNodeInfos"]
    NetJson=param["NetJson"]
    artstudio_excutor_path=param["artstudio_excutor_path"]
    mixQuantJsonHub=param["mixQuantJsonHub"]
    ini_validpath=ini_inpath[:-4]+"_fastprocessing.ini"
    dst_bwjson=os.path.join(mixQuantJsonHub, get_ini_name(ini_inpath)+"_per_block.json")
    for _index, blkName in enumerate(BlkNodeInfos.keys()):
        BlkInfos={}
        BlkInfos[blkName]=BlkNodeInfos[blkName]
        NewNetJson=set_bwjson_fast(NetJson,BlkInfos)
        #NewNetJson=precisionModify(NewNetJson)
        save_json(NewNetJson, dst_bwjson)
        set_ini_bw_json(ini_inpath, ini_validpath, dst_bwjson)
        run_artstudio(artstudio_excutor_path, ini_validpath, output_base_dir)
        output_dir=os.path.join(output_base_dir,get_ini_name(ini_validpath))
        _cycle,_rmse=get_total_cycle_mse(output_dir)
        data={"sum_cycle":_cycle, "sum_rmse":_rmse}
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


