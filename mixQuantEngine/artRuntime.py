# from pandas import read_csv
import os
import json
import csv
import subprocess
import subprocess
from shutil import copyfile

def remove_dir(dir):
    if os.path.isdir(dir):
        dirlists=os.listdir(dir)
        if not dirlists:
            if os.path.exists(dir):
                os.removedirs(dir)
        else:
            for file in dirlists:
                remove_dir(os.path.join(dir, file))
    else:
        if os.path.exists(dir):
            os.remove(dir)

def create_dir(dir):
    try:
        os.mkdir(dir)
    except:
        print(dir," exist!")
        pass

def get_ini_name(dir):
    return dir.split("/")[-1].split(".")[0]

def run_artstudio(art_path, ini_file, out_dir, UseSrc, run_compiler=True):
    #command_dir = "cd {}".format(art_path)
    if UseSrc:
        PYTHONPATH = os.path.join(art_path, "art_ort/build/linux/python37_build/cpu/")
        command_str = "export PATHONPATH={};cd {};python {} -i {} -o {}".format(PYTHONPATH, art_path, 'acnn.py', ini_file, out_dir)
    else:
        command_str = "{} --ini {} -o {}".format(art_path, ini_file, out_dir)
    if not run_compiler:
        command_str += " --nonpubin"
    print("===========================================================")
    print(command_str)
    subprocess.call(command_str, shell=True)  
    # subprocess.run(command_str, shell=True)
    # os.system(command_str)

def read_cycle(cycleFile):
    csvFile = open(cycleFile, "r")
    reader = csv.reader(csvFile)
    total_cycles=0
    for item in reader:
        if reader.line_num == 1:
            continue
        # print(type(item[0]),type("int_register_total_cycle")
        if str("total_cycle") in str(item[0]):
            for cycle in item[1:]:
                # print(cycle)
                total_cycles+=int(cycle)     
            break
    # exit(1)
    csvFile.close()
    return total_cycles
    

def read_mse(mse_file, dump_layer_names):
    csvFile = open(mse_file, "r")
    reader = csv.reader(csvFile)
    precision = {}
    output_rmse=0.0
    for item in reader:
        if reader.line_num == 1:
            continue
        #for dump_layer_name in dump_layer_names:
        if item[0] in dump_layer_names:
            output_rmse+=float(item[-2])
    csvFile.close()
    return output_rmse, precision

def read_cossim(mse_file, dump_layer_names):
    csvFile = open(mse_file, "r")
    reader = csv.reader(csvFile)
    output_cossim=0.0
    for item in reader:
        if reader.line_num == 1:
            continue
        #for dump_layer_name in dump_layer_names:
        if item[0] in dump_layer_names:
            output_cossim+=float(item[-1])
    csvFile.close()
    return output_cossim

# def get_output_tensors(json_file):
#     file = open(json_file,'r',encoding='utf-8')
#     netJson = json.load(file)
#     tensor_name=[]
#     tensor_num=netJson["output_tensor_number"]
#     for i in range(tensor_num):
#         tensor_name.append(netJson["output_tensor_"+str(i)]["tensor_name"])
#     return tensor_name
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

def get_msefile_path(output_dir):
    mse_folder=os.path.join(output_dir,"mse_result")
    mse_file_list=os.listdir(mse_folder)
    for mse_file in mse_file_list:
        if ".csv" in mse_file:
            return os.path.join(mse_folder,mse_file)
    return None

def get_average_cossim(output_dir, tensor_name):
    mse_folder=os.path.join(output_dir,"mse_ori_result")
    if not os.path.exists(mse_folder):
        mse_folder=os.path.join(output_dir,"mse_result")
    mse_file_list=os.listdir(mse_folder)
    sum_cossim=0.0
    for mse_file in mse_file_list:
        if ".csv" in mse_file:
            _sum_cosisim=read_cossim(os.path.join(mse_folder,mse_file), tensor_name)
            sum_cossim+=_sum_cosisim
    return sum_cossim / len(tensor_name)         

def get_total_cycle_mse(output_dir, tensor_name, dump_cycle=True):
    try:
        init_total_cycle = 0
        if dump_cycle:
            cycleFile=os.path.join(output_dir,"bin_hex/LayerPerformance.csv")
            init_total_cycle = read_cycle(cycleFile)

        # output_json_file=os.path.join(output_dir,"bin_hex/output.json")
        # tensor_name=get_output_tensors(output_json_file)

        mse_folder=os.path.join(output_dir,"mse_ori_result")
        if not os.path.exists(mse_folder):
            mse_folder=os.path.join(output_dir,"mse_result")
        mse_file_list=os.listdir(mse_folder)
        sum_rmse=0.0
        for mse_file in mse_file_list:
            if ".csv" in mse_file:
                _sum_rmse, _precision=read_mse(os.path.join(mse_folder,mse_file), tensor_name)
                sum_rmse+=_sum_rmse

        return init_total_cycle,sum_rmse/len(tensor_name)
    except:
        init_total_cycle = 0
        if dump_cycle:
            cycleFile=os.path.join(output_dir,"bin_hex/LayerPerformance.csv")
            init_total_cycle = read_cycle(cycleFile)

        # output_json_file=os.path.join(output_dir,"bin_hex/output.json")
        # tensor_name=get_output_tensors(output_json_file)

        mse_folder=os.path.join(output_dir,"mse_result")
        if not os.path.exists(mse_folder):
            mse_folder=os.path.join(output_dir,"mse_ori_result")
        mse_file_list=os.listdir(mse_folder)
        sum_rmse=0.0
        for mse_file in mse_file_list:
            if ".csv" in mse_file:
                _sum_rmse, _precision=read_mse(os.path.join(mse_folder,mse_file), tensor_name)
                sum_rmse+=_sum_rmse

        return init_total_cycle,sum_rmse/len(tensor_name)

def setOptModelOfOnnx(iniPath, outBaseDir, optModelPath, optModelName):
    file = open(iniPath,'r',encoding='utf-8')
    iniJson = json.load(file)
    outOptDir=os.path.join(outBaseDir, "opt-model")
    create_dir(outOptDir)
    outOptFileDir=os.path.join(outOptDir, optModelName)
    copyfile(optModelPath, outOptFileDir)
    iniDstPath=os.path.join(outOptDir, get_ini_name(iniPath)+".ini")
    iniJson["netConfig"]["net_path"]=outOptFileDir
    if iniJson["netConfig"]["framework"]=="caffe":
        iniJson["netConfig"]["framework"]="onnx"
        del iniJson["netConfig"]["weight_path"]
    fileOut=open(iniDstPath,"w",encoding='utf-8')
    json.dump(iniJson,fileOut,ensure_ascii=False,indent=1)
    file.close()
    fileOut.close()
    return iniDstPath


def set_ini_bw_json(input_ini,out_ini,json_path):
    file = open(input_ini,'r',encoding='utf-8')
    netJson = json.load(file)
    netJson["netConfig"]["bw_json"]=json_path
    netJson["netConfig"]["do_c_model_verify"]=False
    netJson["netConfig"]["dump_result_en"] = False
    netJson["netConfig"]["quantize_en"]=True
    fileOut=open(out_ini,"w",encoding='utf-8')
    json.dump(netJson,fileOut,ensure_ascii=False,indent=1)

def set_ini_bit(input_ini,out_ini,bw):
    file = open(input_ini,'r',encoding='utf-8')
    netJson = json.load(file)
    if "buildConfig" not in netJson:
        netJson["buildConfig"]={}
    netJson["buildConfig"]["logLevel"]=0
    netJson["netConfig"]["precision"]=bw
    netJson["netConfig"]["bw_json"]=""
    netJson["netConfig"]["do_c_model_verify"]=False
    netJson["netConfig"]["dump_result_en"] = False
    netJson["netConfig"]["quantize_en"]=True
    fileOut=open(out_ini,"w",encoding='utf-8')
    json.dump(netJson,fileOut,ensure_ascii=False,indent=1)

def load_json(json_path):
    file = open(json_path,'r',encoding='utf-8')
    netJson = json.load(file)
    return netJson

def save_json(netJson,json_path):
    fileOut=open(json_path,"w",encoding='utf-8')
    json.dump(netJson,fileOut,ensure_ascii=False,indent=3)
