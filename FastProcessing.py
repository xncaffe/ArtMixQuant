import numpy as np
import os
import csv
import copy
import onnx
# import multiprocessing 
from mixQuantEngine.fastEncoderDecoder import *
from mixQuantEngine.fastMixCalculate import *
from mixQuantEngine.v2netEncoder import *
from mixQuantEngine.artRuntime import *
#from threading import Thread
#from NormalProcessing import *
BENCH_INIT=1

def get_maxlen_blk(BlkNodeInfos, param):
    onnx_model = param["opt_model"]
    initialLists = param["initialLists"]
    setInitJson = param["setInitJson"]
    blkLens=[]
    setBlkDicts={}
    for _index, blkName in enumerate(BlkNodeInfos.keys()):
        BlkNodes=BlkNodeInfos[blkName]
        setBlkList=[]
        for idx, nodeName in enumerate(BlkNodes):
            output_tensor_names = BlkNodes[nodeName]["output_tensor_names"]
            for output_tensor_name in output_tensor_names:
                correspondInitials=find_initializer_from_node(output_tensor_name, onnx_model, initialLists)
                for setInitialName in correspondInitials:
                    if setInitialName not in setBlkList:
                        setBlkList.append(setInitialName)
            insertNodeName = nodeName + "_{}".format(BlkNodes[nodeName]["operator_type"])
            if insertNodeName in setInitJson.keys():
                setBlkList.append(insertNodeName)   
        blkLens.append(len(setBlkList))
        setBlkDicts[blkName]=setBlkList
    return max(blkLens), setBlkDicts

def mixprecision_search(param, record_name):
    BlkNodeInfos=param["BlkNodeInfos"] 
    initPerformPerBlk=loop_setprecision_perblk(param)
    qsortPerformPerBlk=qsort_rmse(initPerformPerBlk)
    outData,Id,PerformStackBlk =minquant_stack_eachblk(param, qsortPerformPerBlk)
    if outData['sum_cycle'] < param["targetCycle"] and outData['sum_rmse'] < param["targetRmse"]:
        print("############################################################################################")
        print("The mixed quantification is completed, and the target rmse and cycle have been reached!")
        print("Writing mix quant results..........................................")
        print("############################################################################################")
    info_line = ['mix', '{}'.format(outData["sum_cycle"]), '%.8f'%outData["sum_rmse"], '%.6f'%outData["ave_cossim"]]
    line_void = ""
    blkTitle = ["node_priority"]
    subBlkTitle=["Id","Block", "Nodes"]
    maxLen, setBlkDicts=get_maxlen_blk(BlkNodeInfos, param)
    for i in range(maxLen+2-len(subBlkTitle)):
        subBlkTitle.append("")
    subBlkTitle.append("Used")
    subBlkTitle.append("cycle")
    subBlkTitle.append("ave_rmse")
    subBlkTitle.append("ave_cossim")
    with open(record_name,'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(info_line)
        for i in range(2):
            f_csv.writerow(line_void)
        f_csv.writerow(blkTitle)
        f_csv.writerow(subBlkTitle)
        for _index, blkName in enumerate(qsortPerformPerBlk.keys()):
            blkNodesArray=[str(_index), blkName]
            blkTensorLists=setBlkDicts[blkName]
            for _id, op_name in enumerate(blkTensorLists):
                blkNodesArray.append(op_name)
            if len(blkNodesArray)<(maxLen+2):
                for i in range(maxLen+2-len(blkNodesArray)):
                    blkNodesArray.append("")
            if _index<=Id:
                blkNodesArray.append("yes")
            else:
                blkNodesArray.append("no")
            if _index > 0 and _index in PerformStackBlk:
                blkNodesArray.append("%d"%PerformStackBlk[_index]["sum_cycle"])
                blkNodesArray.append("%.8f"%PerformStackBlk[_index]["sum_rmse"])
                blkNodesArray.append("%.6f"%PerformStackBlk[_index]["ave_cossim"])
            f_csv.writerow(blkNodesArray)
    print("Fast mix quant search completed, please confirm the performance.")
    return outData

def fast_process(args):
    artstudio_root_path=args.a
    UseSrc=args.SRC
    #maskDump=args.maskDump
    if UseSrc:
        artstudio_excutor_path=os.path.join(artstudio_root_path,"src/generator")
    else:
        artstudio_excutor_path=os.path.join(artstudio_root_path,"cmd/acnn")
    ini=args.ini
    output_base_dir=args.o
    
    # run all 16 bit result
    ini_16bit_path=os.path.join(output_base_dir, get_ini_name(ini)+"_16.ini")
    output_16bit_dir=os.path.join(output_base_dir,get_ini_name(ini_16bit_path))
    # create_dir(output_16bit_dir)
    set_ini_bit(ini,ini_16bit_path,16)
    # if BENCH_INIT:
    #     # run_artstudio(artstudio_excutor_path,ini_16bit_path,output_base_dir, UseSrc, maskDump)
    #     run_artstudio(artstudio_excutor_path,ini_16bit_path,output_base_dir, UseSrc)
    #get base informations
    opt_model_path, opt_model_name=get_optmodel_path(output_16bit_dir)
    opt_model=onnx.load(opt_model_path)
    bw_json=os.path.join(output_16bit_dir, "final_operator_internal_bw.json")
    initJson=parse_json(bw_json)
    NetJson, initialLists=get_connection_net(initJson, opt_model)
    #save_json(NetJson, os.path.join(args.o, "error_repair.json"))
    netOutTensors=get_output_tensors(NetJson)
    #read 16bit cycle and rmse
    total_cycle_16bit,sum_rmse_16bit=get_total_cycle_mse(output_16bit_dir, netOutTensors)
    ave_cossim_16bit=get_average_cossim(output_16bit_dir, netOutTensors)
    
    # run all 8bit result
    ini_8bit_path=os.path.join(output_base_dir, get_ini_name(ini)+"_8.ini")
    output_8bit_dir=os.path.join(output_base_dir,get_ini_name(ini_8bit_path))
    # create_dir(output_8bit_dir)
    set_ini_bit(ini,ini_8bit_path,8)
    # if BENCH_INIT:      
    #     # run_artstudio(artstudio_excutor_path,ini_8bit_path,output_base_dir, UseSrc, maskDump)
    #     run_artstudio(artstudio_excutor_path,ini_8bit_path,output_base_dir, UseSrc)
    # get cycle and mse
    total_cycle_8bit,sum_rmse_8bit=get_total_cycle_mse(output_8bit_dir, netOutTensors)
    ave_cossim_8bit=get_average_cossim(output_8bit_dir, netOutTensors)

    print("========================FINISH 8BIT 16BIT INIT RUN====================================")
    print("8  bit cycle:{} avermse:{}, avecossim:{}".format(total_cycle_8bit,'%.8f'%sum_rmse_8bit, '%.6f'%ave_cossim_8bit))
    print("16 bit cycle:{} avermse:{}, avecossim:{}".format(total_cycle_16bit,'%.8f'%sum_rmse_16bit, '%.6f'%ave_cossim_16bit))
    print("===============================START MIX QUANT========================================")

    target_cycle_ratio=float(args.cr)
    target_rmse_ratio=float(args.mr)
    target_rmse=(sum_rmse_8bit-sum_rmse_16bit)*(1.0-target_rmse_ratio)+sum_rmse_16bit
    target_cycle=int((1-target_cycle_ratio)*(total_cycle_16bit -total_cycle_8bit)+total_cycle_8bit)

    BlkNodeInfos=get_block_node_infos(NetJson)
    if args.UseOpt:
        ini_mix_path=setOptModelOfOnnx(ini_16bit_path, output_base_dir, opt_model_path, opt_model_name)
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
    header = ['bit_name','cycle','ave_rmse', "ave_cossim"]
    info_line = ['8bit', '{}'.format(total_cycle_8bit), '%.8f'%sum_rmse_8bit, '%.6f'%ave_cossim_8bit]
    with open(record_name,'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerow(info_line)
    info_line = ['16bit', '{}'.format(total_cycle_16bit), '%.8f'%sum_rmse_16bit, '%.6f'%ave_cossim_16bit]
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
    param["setInitJson"]=initJson
    param["initialLists"]=initialLists
    param["opt_model"]=opt_model
    param["UseSrc"]=UseSrc
    param["netOutTensors"]=netOutTensors
    #param["maskDump"]=maskDump
    outData=mixprecision_search(param, record_name)
    
    remove_dir(ini_16bit_path)
    remove_dir(ini_8bit_path)
    ini_validpath=ini[:-4]+"_fastprocessing.ini"
    remove_dir(ini_validpath)

    print("********************************* Fast mixed precision quantization results **********************************")
    print("Only  16 bit -> ave rmse: %.8f || " % sum_rmse_16bit + "sum cycles: %d || " % total_cycle_16bit + "ave cossim: %.6f" % ave_cossim_16bit)
    print("Only   8 bit -> ave rmse: %.8f || " % sum_rmse_8bit + "sum cycles: %d || " % total_cycle_8bit + "ave_cossim: %.6f" % ave_cossim_8bit)
    print("Target   inf -> ave rmse: %.8f || " % target_rmse + "sum cycles: %d || " % target_cycle)
    print("Mix    Quant -> ave rmse: %.8f || " % outData["sum_rmse"] + "sum cycles: %d || " % outData["sum_cycle"] + "ave_cossim: %.6f" % outData["ave_cossim"])
    print("**************************************************************************************************************")
    print("Finish Operate!")

    
