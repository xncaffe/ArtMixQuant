import argparse
from cv2 import split
import numpy as np
import os
import csv
import copy
# import multiprocessing 
from mixQuantEngine.fastEncoderDecoder import *
from mixQuantEngine.fastMixCalculate import *
from threading import Thread
from NormalProcessing import *

def get_maxlen_blk(BlkNodeInfos):
    blkLen=[]
    for _index, blkName in enumerate(BlkNodeInfos.keys()):
        BlkNodes=BlkNodeInfos[blkName]
        blkLen.append(len(BlkNodes))
    return max(blkLen)

def mixprecision_search(param, record_name):
    BlkNodeInfos=param["BlkNodeInfos"]  
    initPerformPerBlk=loop_setprecision_perblk(param)
    qsortPerformPerBlk=qsort_rmse(initPerformPerBlk)
    outData,Id=minquant_stack_eachblk(param, qsortPerformPerBlk)
    info_line = ['mix', '{}'.format(outData["sum_cycle"]), '{}'.format(outData["sum_rmse"])]
    line_void = ""
    blkTitle = ["node_priority"]
    subBlkTitle=["Id","Block", "Nodes"]
    maxLen=get_maxlen_blk(BlkNodeInfos)
    for i in range(maxLen+2-len(subBlkTitle)):
        subBlkTitle.append("")
    subBlkTitle.append("Used")
    with open(record_name,'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(info_line)
        for i in range(2):
            f_csv.writerow(line_void)
        f_csv.writerow(blkTitle)
        f_csv.writerow(subBlkTitle)
        for _index, blkName in enumerate(qsortPerformPerBlk.keys()):
            blkNodesArray=[str(_index), blkName]
            blkNodes=BlkNodeInfos[blkName]
            for _id, op_name in enumerate(blkNodes.keys()):
                blkNodesArray.append(op_name)
            if len(blkNodesArray)<(maxLen+2):
                for i in range(maxLen+2-len(blkNodesArray)):
                    blkNodesArray.append("")
            if _index<=Id:
                blkNodesArray.append("yes")
            else:
                blkNodesArray.append("no")
            f_csv.writerow(blkNodesArray)
    print("Fast mix quant search completed, please confirm the performance.")
    return outData

def fast_process(args):
    artstudio_root_path=args.a
    artstudio_excutor_path=os.path.join(artstudio_root_path,"art.studio/ArtStudio")
    ini=args.ini
    output_base_dir=args.o
    # run all 8bit result
    ini_8bit_path=ini[:-4]+"_8hp.ini"
    output_8bit_dir=os.path.join(output_base_dir,get_ini_name(ini_8bit_path))
    # create_dir(output_8bit_dir)
    set_ini_bit(ini,ini_8bit_path,0)
    if BENCH_INIT:      
        run_artstudio(artstudio_excutor_path,ini_8bit_path,output_base_dir)
    # get cycle and mse
    total_cycle_8bit,sum_rmse_8bit=get_total_cycle_mse(output_8bit_dir)

    # run all 16 bit result
    ini_16bit_path=ini[:-4]+"_16.ini"
    output_16bit_dir=os.path.join(output_base_dir,get_ini_name(ini_16bit_path))
    # create_dir(output_16bit_dir)
    set_ini_bit(ini,ini_16bit_path,16)
    if BENCH_INIT:
        run_artstudio(artstudio_excutor_path,ini_16bit_path,output_base_dir)
    total_cycle_16bit,sum_rmse_16bit=get_total_cycle_mse(output_16bit_dir)

    print("========================FINISH 8BIT 16BIT INIT RUN====================================")
    print("8hp bit cycle:{} rmse:{}".format(total_cycle_8bit,sum_rmse_8bit))
    print("16  bit cycle:{} rmse:{}".format(total_cycle_16bit,sum_rmse_16bit))
    print("===============================START MIX QUANT=======================================")

    target_cycle_ratio=float(args.cr)
    target_rmse_ratio=float(args.mr)
    target_rmse=(sum_rmse_8bit-sum_rmse_16bit)*(1.0-target_rmse_ratio)+sum_rmse_16bit
    target_cycle=int((1-target_cycle_ratio)*(total_cycle_16bit -total_cycle_8bit)+total_cycle_8bit)

    bw_json=os.path.join(output_16bit_dir, "final_operator_internal_bw.json")
    NetJson=parse_json(bw_json)
    BlkNodeInfos=get_block_node_infos(NetJson)
    if args.UseOpt:
        ini_mix_path=setOptModelOfOnnx(ini_16bit_path, output_base_dir)
    else:
        ini_mix_path=ini_16bit_path
    
    mixQuantJsonHub=os.path.join(output_base_dir,"mixQuantJsonHub")
    if os.path.exists(mixQuantJsonHub):
        remove_dir(mixQuantJsonHub)
    create_dir(mixQuantJsonHub)

    record_name="output_"+str(args.s)+".csv"
    record_name = os.path.join(output_base_dir, record_name)
    if os.path.exists(record_name):
        os.remove(record_name)
    header = ['bit_name','cycle','rmse']
    info_line = ['8(hp)bit', '{}'.format(total_cycle_8bit), '{}'.format(sum_rmse_8bit)]
    with open(record_name,'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerow(info_line)
    info_line = ['16bit', '{}'.format(total_cycle_16bit), '{}'.format(sum_rmse_16bit)]
    with open(record_name,'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(info_line)

    param={}
    param["artstudio_excutor_path"]=artstudio_excutor_path
    param["ini_mix_path"]=ini_mix_path
    param["output_base_dir"]=output_base_dir
    param["NetJson"]=NetJson
    param["BlkNodeInfos"]=BlkNodeInfos
    param["targetRmse"]=target_rmse
    param["targetCycle"]=target_cycle
    param["mixQuantJsonHub"]=mixQuantJsonHub
    outData=mixprecision_search(param, record_name)

    print("************************* Fast mixed precision quantization results ****************************")
    print("Only  16 bit -> sum rmse: %.6f || " % sum_rmse_16bit + "sum cycles: %d" % total_cycle_16bit)
    print("Only 8hp bit -> sum rmse: %.6f || " % sum_rmse_8bit + "sum cycles: %d" % total_cycle_8bit)
    print("Target   inf -> sum rmse: %.6f || " % target_rmse + "sum cycles: %d" % target_cycle)
    print("Mix    Quant -> sum rmse: %.6f || " % outData["sum_rmse"] + "sum cycles: %d" % outData["sum_cycle"])
    print("************************************************************************************************")
    print("Finish Operate!")

    
