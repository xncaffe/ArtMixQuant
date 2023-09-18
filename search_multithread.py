import argparse
from cv2 import split
import numpy as np
import os
import csv
import math
import copy
import matplotlib.pyplot as plt
# import multiprocessing 
from threading import Thread

import scipy as sp
from mixQuantEngine.mixQuantCorrecter import *
from mixQuantEngine.artRuntime import *
from mixQuantEngine.netEncoderDecoder import *
from mixQuantModel.DDEController import *
BENCH_INIT=1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", default="/workspace/nxu/ArtStudio/artfast-dev-0610/", help="artstudio root path")
    parser.add_argument("-ini", default="/workspace/nxu/ArtStudio/artfast-dev-0610/workspace/aobi_Xs2/model/aobi_Xs2.ini", help="artstudio ini path")
    parser.add_argument("-o", default="/workspace/nxu/ArtStudio/artfast-dev-0610/workspace/aobi_Xs2/output", help="artstudio output dir")
    parser.add_argument("-w", default=10, help="multi process workers")
    parser.add_argument("-m", default=20, help="sample group")
    parser.add_argument("-s", default=0, help="test_slave")
    parser.add_argument("-r", default=10000, help="search round")
    parser.add_argument("-rd", default=0, help="random bais, set 16 or 8 or 0")
    parser.add_argument("-d", default=1, help="mix precision set direction, set 1 or 0")
    parser.add_argument("--PF", action='store_true', help="mse penalty factor or not, valid when mt=1")
    parser.add_argument("--UseOpt", action='store_true', help="use opt-model or no when mix quant search. Only support ONNX!")
    parser.add_argument("-xovr", default=0.1, help="xovr")
    parser.add_argument("-mp", default=0.85, help="mp")
    args = parser.parse_args()
    return args

def get_linear_score(val,min_val,max_val):
    if val>max_val:
        val=max_val
    if val<min_val:
        val=min_val
    return (val-min_val)/(max_val-min_val)

def get_reward(cycle,rmse,total_cycle_8bit,sum_rmse_8bit,total_cycle_16bit,sum_rmse_16bit,penalty=False):
    rmse_score=(1-get_linear_score(rmse,sum_rmse_16bit,sum_rmse_8bit))
    if rmse_score>0.5:
        rmse_score=rmse_score*3
    else:
        if penalty:
            if rmse_score>0.1:
                rmse_score=rmse_score*2
            else:
                rmse_score=rmse_score*0.5
        else:
            rmse_score=rmse_score*0.6
    cycle_score=get_linear_score(cycle,total_cycle_8bit,total_cycle_16bit)
    if cycle_score>0.9:
        cycle_score=cycle_score*10
    elif cycle_score>0.7:
        cycle_score=cycle_score*3
    else:
        cycle_score=cycle_score*0.5

    return rmse_score-cycle_score


def studio_processing_worker(param):
    artstudio_excutor_path=param["artstudio_excutor_path"]
    ini_test_path=param["ini_test_path"]
    output_base_dir=param["output_base_dir"]
    test_id=param["test_id"]
    current_param=param["current_param"]
    run_artstudio(artstudio_excutor_path,ini_test_path,output_base_dir)
    output_test_dir=os.path.join(output_base_dir,get_ini_name(ini_test_path))
    total_cycle_test,sum_rmse_test=get_total_cycle_mse(output_test_dir)
    return_val={"test_id":test_id,"total_cycle_test":total_cycle_test,"sum_rmse_test":sum_rmse_test,"current_param":current_param}
    return return_val

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args       

    def run(self): 
        self.result = self.func(self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
def draw_state_fig(x, y, xlabel, ylabel, lc, name, wsplit_end=0, wsplit_start=False):
    fig=plt.figure()
    end_n=0-wsplit_end
    if wsplit_start:  
        plt.plot(x[:2], y[:2], 'ro')
        plt.plot(x[:2], y[:2], 'r')
        if wsplit_end>0:
            plt.plot(x[2:end_n],y[2:end_n], lc)
            plt.plot(x[end_n:],y[end_n:],'yo')
        else:
            plt.plot(x[2:],y[2:], lc)
    else:
        if wsplit_end>0:
            plt.plot(x[:end_n], y[:end_n], lc)
            plt.plot(x[end_n:], y[end_n:], 'yo')
        else:
            plt.plot(x, y, lc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    fig.savefig(name)
    plt.close(fig)


def main():
    args = parse_args()
    artstudio_root_path=args.a
    if int(args.m)<int(args.w):
        args.w=args.m
    if int(args.m)%int(args.w)!=0:
        args.m=int(args.m)//int(args.w)*int(args.m)
    artstudio_excutor_path=os.path.join(artstudio_root_path,"art.studio/ArtStudio")
    ini=args.ini
    output_base_dir=args.o
    # run all 8bit result
    ini_8bit_path=ini[:-4]+"_8.ini"
    output_8bit_dir=os.path.join(output_base_dir,get_ini_name(ini_8bit_path))
    # create_dir(output_8bit_dir)
    set_ini_bit(ini,ini_8bit_path,8)
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
    print("8 bit cycle:{} rmse:{}".format(total_cycle_8bit,sum_rmse_8bit))
    print("16bit cycle:{} rmse:{}".format(total_cycle_16bit,sum_rmse_16bit))
    print("")
    print("")
    print("===============================START MIX QUANT=======================================")

    record_name="output_"+str(args.s)+".csv"
    if os.path.exists(record_name):
        os.remove(record_name)
    header = ['case','cycle','rmse','reward']
    info_line = ['8bit', '{}'.format(total_cycle_8bit), '{}'.format(sum_rmse_8bit)]
    with open(record_name,'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerow(info_line)
    info_line = ['16bit', '{}'.format(total_cycle_16bit), '{}'.format(sum_rmse_16bit)]
    with open(record_name,'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(info_line)
    
    if int(args.d)==1:
        ini_mix_path = ini_16bit_path
    else:
        ini_mix_path = ini_8bit_path
    if args.UseOpt:
        ini_mix_path=setOptModelOfOnnx(ini_mix_path, output_base_dir)
    mixQuantJsonHub=os.path.join(output_base_dir,"mixQuantJsonHub")
    if os.path.exists(mixQuantJsonHub):
        remove_dir(mixQuantJsonHub)
    create_dir(mixQuantJsonHub)
    _JsonDirSturct={"ini_mix_path":ini_mix_path,"mixJsonDir":mixQuantJsonHub}
    ori_mixQuantJson=os.path.join(output_base_dir,get_ini_name(ini_mix_path)+"/final_operator_internal_bw.json")
    netJson=load_json(ori_mixQuantJson)
    precision_code_array=mixQuantJson2SearchArray(netJson)

    print(precision_code_array)
    agent=DiscreteDifferentialEvolutionController(MIND=int(args.m),M=float(args.mp),XOVR=float(args.xovr),
                SEARCH_SPACE=mixQuantJson2SearchArray(netJson),bigger_is_better=True,random_bais=int(args.rd))
    # result_recorder={}
    x_iter = []
    y_reward = []
    x_cycle = [total_cycle_8bit,total_cycle_16bit]
    y_rmse = [sum_rmse_8bit, sum_rmse_16bit]
    reward_curve="reward_case.png"
    cycle_rmse_dot="cycle_rmse_vary.png"
    draw_state_fig(np.array(x_cycle), np.array(y_rmse), "Cycle", "rmse", "ro", cycle_rmse_dot)
    agent.init_performance.append({"cycle":total_cycle_8bit, "rmse":sum_rmse_8bit})
    agent.init_performance.append({"cycle":total_cycle_16bit, "rmse":sum_rmse_16bit})

    for _r in range(0,int(args.r)//int(args.w)):
        jobs = []
        for _w in range(int(args.w)):
            test_id=int(args.w)*_r+_w
            print(test_id)
            current_param=agent.get_newparam()
            for i in range(len(precision_code_array)):
                if(current_param[i]==1):
                    precision_code_array[i]=16
                else:
                    precision_code_array[i]=8
            #precision_code_array[-1]=16
            newNetJson=Array2mixQuantJson(netJson,precision_code_array)
            newNetJson=precisionModify(newNetJson)
            newNetJsonPath=os.path.join(mixQuantJsonHub,get_ini_name(ini_mix_path))+"_"+str(test_id)+".json"
            save_json(newNetJson,newNetJsonPath)
            ini_test_path=ini[:-4]+"-process-worker"+str(_w)+".ini"
            set_ini_bw_json(ini_mix_path,ini_test_path,newNetJsonPath)
            # multiprocess
            param={}
            param["artstudio_excutor_path"]=artstudio_excutor_path
            param["ini_test_path"]=ini_test_path
            param["output_base_dir"]=output_base_dir
            param["test_id"]=test_id
            param["current_param"]=current_param
            _thread=MyThread(studio_processing_worker, param)
            # process = multiprocessing.Process(target=studio_processing_worker, args=(param, process_queue))

            jobs.append(_thread)
            _thread.start()

        for process in jobs:
            process.join()
        # for process in jobs:
        #     print(process.get_result())
        process_results=[j.get_result() for j in jobs]
        # feed back reward
        for _w in range(int(args.w)):
            test_id=int(args.w)*_r+_w
            found_flg=False
            for result in process_results:
                # print(result)
                if result["test_id"]==test_id:
                    total_cycle_test=result["total_cycle_test"]
                    sum_rmse_test=result["sum_rmse_test"]
                    current_param=result["current_param"]
                    found_flg=True
                    break
            if found_flg==False:
                print("ERROR, RESULT IN MULTIPROCESS NOT FOUND!")
                assert(0)
            reward_value=get_reward(total_cycle_test,sum_rmse_test,total_cycle_8bit,sum_rmse_8bit,total_cycle_16bit,sum_rmse_16bit,penalty=args.PF)
            # print(current_param)
            if(reward_value != 0):
                relate_cycle=float(total_cycle_test)/float(total_cycle_16bit)
                relate_rmse=float(sum_rmse_test)/float(sum_rmse_8bit)
                DistanceTo0 = np.sqrt(np.square(relate_cycle) + np.square(relate_rmse))
                
                agent.update_reward(current_param,reward_value,total_cycle_test,sum_rmse_test,DistanceTo0,test_id,_JsonDirSturct)
            
            info_line = ['test {} round'.format(test_id), '{}'.format(total_cycle_test), '{}'.format(sum_rmse_test), '{}'.format(reward_value)]
            with open(record_name, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(info_line)
            x_iter.append(test_id)
            y_reward.append(reward_value)
            draw_state_fig(np.array(x_iter), np.array(y_reward), "case_iter", "reward", "b", reward_curve)
            x_cycle.append(total_cycle_test)
            y_rmse.append(sum_rmse_test)
            wBestX_cycle=copy.deepcopy(x_cycle)
            wBestY_rmse=copy.deepcopy(y_rmse)
            for _best_record in agent.best_record:
                wBestX_cycle.append(_best_record["cycle"])
                wBestY_rmse.append(_best_record["rmse"])
            draw_state_fig(np.array(wBestX_cycle),np.array(wBestY_rmse),"Cycle","rmse","bo",
                            cycle_rmse_dot,len(agent.best_record),wsplit_start=True)            

    info_line=''
    best_header = ['best_id','cycle','rmse','reward','relDistance']
    with open(record_name, 'a+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(info_line)
        f_csv.writerow(best_header)
    for _best_record in agent.best_record:
        info_line=[str(_best_record["test"]), str(_best_record["cycle"]),str(_best_record["rmse"]),str(_best_record["reward"]),str(_best_record['relDistance'])]
        with open(record_name, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(info_line)


if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main()
