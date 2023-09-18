import os
import time
import threading
import psutil
import csv
import copy
import json
import subprocess
from shutil import rmtree

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("[ArtMixQuant]")

def real_time_get_peak_wset(params_dict: dict):
    main_pid = params_dict['main_pid']
    try:
        main_process = psutil.Process(main_pid)
    except:
        return
    cur_peak_size = main_process.memory_info().rss
    children_process_list = main_process.children()
    for children_proc in children_process_list:
        cur_peak_size += children_proc.memory_info().rss
        children_process_2level_list = children_proc.children()
        for children_proc_2level in children_process_2level_list:
            cur_peak_size += children_proc_2level.memory_info().rss
    params_dict['peak_wset'] = max(cur_peak_size, params_dict['peak_wset'])


class RealTimePeakWsetThread(threading.Thread):
    def __init__(self, func, args):
        super(RealTimePeakWsetThread, self).__init__()
        self.func = func
        self.args = args
        self.stop_flag = False
        
    def run(self):
        while not self.stop_flag:
            try:
                self.func(self.args)
            except:
                break
            time.sleep(0.05)
    
    def stop(self):
        self.stop_flag = True

def create_dir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except:
        logger.warning(dir + " exist!")
        pass

def remove_dir(dir):
    try:
        if os.path.isdir(dir):
            dirs_list = os.listdir(dir)
            if not dirs_list:
                if os.path.exits(dir):
                    rmtree(dir)
            else:
                for cur_file in dirs_list:
                    remove_dir(os.path.join(dir, cur_file))
            rmtree(dir)
        else:
            if os.path.exists(dir):
                os.remove(dir)
    except:
        logger.warning(dir + " is not exist!")
        pass

def convert_mem_size(size_bytes):
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

    index = 0
    while size_bytes >= 1024. and index < len(units) - 1:
        size_bytes /= 1024.
        index += 1

    return size_bytes, units[index]

def get_ini_name(dir):
    return dir.split("/")[-1].split(".")[0]

def get_excutor_path(artstudio_root_path, use_src=False):
    return os.path.join(artstudio_root_path, 'src/generator') if use_src \
        else os.path.join(artstudio_root_path, 'cmd/acnn')
        
def parse_json(jsonFile):
    fp = open(jsonFile,"r", encoding='utf-8')
    js_infos = json.load(fp)
    fp.close()
    return js_infos

def save_json(netJson,json_path):
    fileOut=open(json_path,"w",encoding='utf-8')
    json.dump(netJson,fileOut,ensure_ascii=False,indent=3)
    
def set_ini_bw_json(input_ini, out_ini, json_path, log_level=0):
    file = open(input_ini,'r',encoding='utf-8')
    netJson = json.load(file)
    netJson["buildConfig"]["logLevel"]=log_level
    netJson["netConfig"]["bw_json"]=json_path
    netJson["netConfig"]["do_c_model_verify"]=False
    netJson["netConfig"]["dump_result_en"] = False
    netJson["netConfig"]["quantize_en"]=True
    fileOut=open(out_ini,"w",encoding='utf-8')
    json.dump(netJson,fileOut,ensure_ascii=False,indent=1)

def set_ini_bit(input_ini, out_ini, bw, log_level=0, opt_model_path=None):
    file = open(input_ini,'r',encoding='utf-8')
    netJson = json.load(file)
    if "buildConfig" not in netJson:
        netJson["buildConfig"]={}
    netJson["buildConfig"]["logLevel"]=log_level
    if opt_model_path is not None:
        if netJson["netConfig"]["framework"]=="caffe":
            netJson["netConfig"]["framework"]="onnx"
            del netJson["netConfig"]["weight_path"]
        netJson["netConfig"]["net_path"] = opt_model_path       
    netJson["netConfig"]["precision"]=bw
    netJson["netConfig"]["bw_json"]=""
    netJson["netConfig"]["do_c_model_verify"]=False
    netJson["netConfig"]["dump_result_en"] = False
    netJson["netConfig"]["quantize_en"]=True
    fileOut=open(out_ini,"w",encoding='utf-8')
    json.dump(netJson,fileOut,ensure_ascii=False,indent=1)
    
def check_output_exist(output_dir):
    if not os.path.exists(output_dir):
        return False
    bw_json_path = os.path.join(output_dir, 'final_operator_internal_bw.json')
    if not os.path.exists(bw_json_path):
        return False
    sub_file_list = os.listdir(output_dir)
    model_list = [sub_file for sub_file in sub_file_list if sub_file[-9:] == '-opt.onnx']
    if not model_list:
        return False
    mse_path = os.path.join(output_dir, 'mse_result')
    if not os.path.exists(mse_path):
        mse_path = os.path.join(output_dir, 'mse_ori_result')
    if not os.path.exists(mse_path):
        return False
    csvfile_list = os.listdir(mse_path)
    csvfile = [file_path for file_path in csvfile_list if file_path[-4:] == '.csv']
    if not csvfile:
        return False
    layer_perform_file = os.path.join(output_dir, 'bin_hex/LayerPerformance.csv')
    if not os.path.exists(layer_perform_file):
        return False
    return True

def read_cycle(cycleFile):
    csvFile = open(cycleFile, "r")
    reader = csv.reader(csvFile)
    total_cycles=0
    for item in reader:
        if reader.line_num == 1:
            continue
        if str("total_cycle") in str(item[0]):
            for cycle in item[1:]:
                total_cycles+=int(cycle)     
            break
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

def get_total_cycle_mse(output_dir, tensor_name, dump_cycle=True):
    init_total_cycle = 0
    if dump_cycle:
        cycleFile=os.path.join(output_dir,"bin_hex/LayerPerformance.csv")
        init_total_cycle = read_cycle(cycleFile)

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

def run_artstudio(art_path, ini_file, out_dir, UseSrc, log_level=0, run_compiler=True):
    #command_dir = "cd {}".format(art_path)
    if UseSrc:
        PYTHONPATH = os.path.join(art_path, "art_ort/build/linux/python37_build/cpu/")
        command_str = "export PATHONPATH={};cd {};python {} -i {} -o {}".format(PYTHONPATH, art_path, 'acnn.py', ini_file, out_dir)
    else:
        command_str = "{} --ini {} -o {}".format(art_path, ini_file, out_dir)
    if not run_compiler:
        command_str += " --nonpubin"
    if log_level < 0:
        command_str += " --ignore_log"
    else:
        print("===========================================================")
        print(command_str)
    subprocess.call(command_str, shell=True)  
    
def find_initializer_from_node(node_name, onnx_model, initials_list):
    correspond_initials = [node_name]
    for idx, node in enumerate(onnx_model.graph.node):
        inputs = node.input
        if node_name in inputs:
            for input in inputs:
                if input in initials_list:
                    correspond_initials.append(input)
    return correspond_initials

def delete_invalid_param(set_json: dict):
    out_set_json=copy.deepcopy(set_json)
    for set_op_name in out_set_json.keys():
        out_set_json_op = copy.deepcopy(out_set_json[set_op_name])
        for param_name in out_set_json_op.keys():
            if param_name == "bit_width":
                continue
            del out_set_json[set_op_name][param_name]
    return out_set_json
    