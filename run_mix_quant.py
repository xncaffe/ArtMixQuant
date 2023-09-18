import argparse
from fast.fastSearchProcess import *
from normal.normalSearchProcess import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", default="./Art.Studio-v0.11/", help="artstudio root path")
    parser.add_argument("-ini", default="./resnet18_onnx.ini", help="artstudio ini path")
    parser.add_argument("-o", default="./mixOutput", help="artstudio output dir")
    # parser.add_argument("-a", type=str, default="/home/nxu/workspace/ArtStudioV2/Art.Studio-dev-0913/", help="artstudio root path")
    # parser.add_argument("-ini", type=str, default="/home/nxu/workspace/model/EfficentViT/EfficentViT_M0_ar9341.ini", help="artstudio ini path")
    # parser.add_argument("-o", type=str, default="./MixOutput", help="artstudio output dir")
    parser.add_argument("-w", type=int, default=1, help="multi process workers")
    parser.add_argument("-pm", type=int, default=0, help="mix precision processing method, 1=standard, 0=fast")
    parser.add_argument("-m", default=20, help="sample group, valid when pm=1")
    parser.add_argument("-s", default=0, help="test_slave, valid when pm=1")
    parser.add_argument("-r", type=int, default=80, help="search round, valid when pm=1")
    parser.add_argument("-rd", type=int, default=0, help="random bais, set 16 or 8 or 0, valid when pm=1")
    parser.add_argument("-d", default=1, help="mix precision set direction, set 1 or 0, valid when pm=1")
    parser.add_argument("--PF", action='store_true', help="mse penalty factor or not, valid when pm=1")
    parser.add_argument("--UseOpt", action='store_true', help="use opt-model or no when mix quant search. Only support ONNX!")
    parser.add_argument("-xovr", type=float, default=0.1, help="xovr, valid when pm=1")
    parser.add_argument("-mp", type=float, default=0.85, help="mp, valid when pm=1")
    parser.add_argument("-cr", default=0.5, help="target cycle ratio, less than 1, valid when pm=0")
    parser.add_argument("-mr", default=0.8, help="target rmse ratio, less than 1, valid when pm=0")
    parser.add_argument("-l", type=int, default=-1, help="log level")
    parser.add_argument("--SRC", action='store_true', help="use src code run acnn.py!")
    #parser.add_argument("--maskDump", action='store_true', help="mask the output of each layer!")
    args = parser.parse_args()
    return args

def mix_init_run(args):
    logger.info("Start automatic mixed-precision initialization ...")
    base_runner = MixBaseRuntime(args.ini, args.a, args.o, args.SRC, args.w, args.l)
    base_runner.check_art_version()
    base_runner.get_before_run_buf()
    base_runner.check_thread_num_cpu(args)
    base_runner.init_file_path()
    base_runner.unpack_base_content()
    base_runner.check_base_performance()
    base_runner.dump_initbit_result()
    base_runner.dump_mix_quant_log()
    base_runner.compute_target_performance(float(args.cr), float(args.mr))
    base_runner.set_opt_model_mix_ini(args.UseOpt)
    base_runner.dump_base_performance()
    base_module = ModuleCodec(base_runner.net_struct_json)
    base_runner.blk_final_nodes_info = base_module.get_block_node_infos()
    base_runner.get_max_length_of_block()
    if base_runner.thread_num > len(base_runner.blk_final_nodes_info.keys()) and args.pm == 0:
        logger.error("The thread_num cannot be greater than the number of modules for fast mode. "
                     + "The current number of network modules is %d, please reset."%len(base_runner.blk_final_nodes_info.keys()))
        assert(0)
    if base_runner.thread_num > 200 and args.pm == 1:
        logger.error("In standard mode, the number of threads should not be greater than 200, please readjust it.")
        assert(0)
    if 200 % base_runner.thread_num > 0:
        logger.warning("In standard mode, the efficiency is highest when the number of threads can be divided evenly by 200.")
    logger.info("Mix quant search initialization finish!")
    return base_runner

def mix_uninit_run(runner_params: MixBaseRuntime):
    logger.info("Mix quant search uninit start ...")
    pure_16bit_outpath_name = os.path.split(runner_params.output_16bit_dir)[-1]
    pure_8bit_outpath_name = os.path.split(runner_params.output_8bit_dir)[-1]
    cur_out_files_list = os.listdir(runner_params.output_base_dir)
    for cur_out_file in cur_out_files_list:
        cur_out_dir = os.path.join(runner_params.output_base_dir, cur_out_file)
        if os.path.exists(cur_out_dir) and not cur_out_file.endswith(
            (".xlsx", "opt-model", "mixQuantJsonHub", pure_16bit_outpath_name, pure_8bit_outpath_name)):
            os.system(f"rm -r {cur_out_dir}")
    
    logger.info("Mix quant search uninit finish!")

def main():
    args = parse_args()
    #args.UseOpt = True
    start_time = time.time()
    
    peak_wset_param = {'main_pid': os.getpid(), 'peak_wset': 0, 'mix_run_flag': True}
    real_time_peak_thread = RealTimePeakWsetThread(real_time_get_peak_wset, peak_wset_param)
    real_time_peak_thread.setDaemon(True)
    real_time_peak_thread.start()
    
    mix_quant_params = mix_init_run(args)  
    method_proc = args.pm
    if method_proc == 0:
        fast_mix_run(mix_quant_params)
    elif method_proc == 1:
        normal_mix_run(mix_quant_params, args)
    else:
        logger.error("The parameter 'pm' setting is invalid, please check!")
    mix_uninit_run(mix_quant_params)
    
    peak_wset_param['mix_run_flag'] = False
    mix_peak_mem, mix_peak_unit = convert_mem_size(peak_wset_param['peak_wset'])
    logger.info('################ Art Mix Quant Peak RSS Memory Estimate #################')
    logger.info('Peak physical memory during search phase: %.6f '%mix_peak_mem + mix_peak_unit)
    logger.info('#########################################################################')
    
    real_time_peak_thread.stop()
    end_time = time.time()
    run_time = (end_time - start_time) / 60.
    logger.info("The mixed precision search time used a total of %.2fmin"%run_time)
    
if __name__ == "__main__":
    main()
    
    