import onnx
import time
import threading
import pandas as pd
from shutil import copyfile
from common.artNetDecode import *

INPUT_OUTPUT_PRECISION_SAME_NODES=["ArtMaxPool", "ArtAveragePool", 
                                   "ArtGlobalMaxPool", "ArtGlobalAveragePool", 
                                   "ArtSoftmaxExp", "CallBack"]

class MixBaseRuntime(object):
    def __init__(self, ini_path, art_path, out_dir, use_src=False, thread_num=1, log_level=-1):
        self.src_ini_path = ini_path
        self.output_base_dir = out_dir
        self.ini_16bit_path = os.path.join(out_dir, get_ini_name(ini_path)+'_16.ini')
        self.ini_8bit_path = os.path.join(out_dir, get_ini_name(ini_path)+'_8.ini')
        self.output_16bit_dir = os.path.join(out_dir, get_ini_name(self.ini_16bit_path))
        self.output_8bit_dir = os.path.join(out_dir, get_ini_name(self.ini_8bit_path))
        self.artstudio_root_path = art_path
        self.art_studio_excutor_path = get_excutor_path(art_path, use_src)
        self.mix_quant_json_hub = os.path.join(out_dir, 'mixQuantJsonHub')
        self.record_excel_path = os.path.join(out_dir, 'output_0.xlsx')
        self.use_src = use_src
        self.thread_num = thread_num
        self.blk_final_nodes_info = None
        self.blk_max_length = 0
        self.set_blocks_dict = {}
        self.sheet_name = 'mix_quant_sheet'
        self.opt_model_path = None
        self.log_level = log_level
    
    def init_file_path(self):
        create_dir(self.output_base_dir)
        remove_dir(self.mix_quant_json_hub)
        if not os.path.exists(self.mix_quant_json_hub):
            create_dir(self.mix_quant_json_hub)
        remove_dir(self.record_excel_path)
    
    def check_thread_num_cpu(self, args):
        kernel_num = os.cpu_count()
        if kernel_num // 2 <= self.thread_num:
            logger.error("The number of configured threads cannot be greater than or equal" 
                         +" to half the number of CPU cores of the current computer.")
            assert(0)
        if args.pm == 1 and self.thread_num > args.m:
            logger.error("In normal mode, the number of threads cannot be greater than the number of populations " 
                         + "(that is, -w cannot be greater than -m, currently w={} and m={})".format(args.w, args.m))
            raise(0)
        if args.pm == 1 and args.m % self.thread_num != 0:
            logger.error("In normal mode, " 
                         + "the number of populations should be an integer multiple of the number of threads " 
                         + "(that is, -m is an integer multiple of -w, currently w={} and m={})".format(args.w, args.m))
            raise(0)
            
    def run_initbit_build(self, bit_num):
        dst_ini_path = self.ini_16bit_path if bit_num == 16 else self.ini_8bit_path
        output_dir = self.output_16bit_dir if bit_num == 16 else self.output_8bit_dir
        if self.opt_model_path is not None:
            set_ini_bit(self.src_ini_path, dst_ini_path, bit_num, log_level=self.log_level, 
                        opt_model_path=self.opt_model_path)
        else:
            set_ini_bit(self.src_ini_path, dst_ini_path, bit_num, log_level=self.log_level)
        if not check_output_exist(output_dir):
            run_artstudio(self.art_studio_excutor_path, dst_ini_path, 
                          self.output_base_dir, self.use_src, log_level=self.log_level)
        else:
            logger.warning("Pure %dbit quantization has been executed before, and the system has found the corresponding file,"%bit_num
                           + "so there is no need to execute it again.")
    
    def check_base_performance(self):
        if self.total_cycle_8bit >= self.total_cycle_16bit:
            logger.error("The calculation cycle of 8bit is greater than or 16bit, which is unreasonable."
                         + "Please feedback to the R&D personnel for inspection")
            assert(0)
        if self.ave_rmse_8bit <= self.ave_rmse_16bit:
            logger.error("The root mean square error of 8bit is less than or equal to 16bit, which is unreasonable." 
                         + "Please feedback to the R&D staff to check")
            assert(0)
        if self.ave_cossim_8bit >= self.ave_cossim_16bit:
            logger.error("The average cosine similarity of 8bit is greater than or equal to 16bit, which is unreasonable,"
                         + "Please feedback to the R&D staff for checking")
            assert(0)
        
    def wait_purebit_build(self):
        logger.info("Waiting for the initialization of pure bit quantization to be completed ...")
        second_time = 0
        while True:
            start_time = time.time()
            cur_8bit_status = check_output_exist(self.output_16bit_dir)
            cur_16bit_status = check_output_exist(self.output_8bit_dir)
            if cur_8bit_status and cur_16bit_status:
                break
            min_time = second_time // 60
            hour_time = min_time // 60
            if hour_time >= 3:
                logger.info("The background pure bit quantization has been running for three hours.")
                logger.info("Please manually execute Art.Studio to ensure that this is normal. If normal, please ignore this warning.")
                logger.info("If it is abnormal, please stop the mixed quantization manually,"
                            +"and then check the abnormal status of pure bit quantization.")
            remain_second = second_time % 60
            remain_min = min_time % 60
            print("\r", end="")
            print("Waiting time: {}h-{}min-{}s".format((str(hour_time)).zfill(2), str(remain_min).zfill(2), str(remain_second).zfill(2)), end="")
            end_time = time.time()
            run_time = end_time - start_time
            time.sleep((1.-run_time) if run_time < 1 else 0)
            second_time += 1  
    
    def thread_run_init8bit_build(self):
        self.run_initbit_build(8)
        
    def thread_run_init16bit_build(self):
        self.run_initbit_build(16)         
    
    def unpack_base_content(self):
        logger.info("Get pure 8bit and 16bit performance data.")
        thread_run8bit = threading.Thread(target=self.thread_run_init8bit_build)
        thread_run8bit.start()
        thread_run16bit = threading.Thread(target=self.thread_run_init16bit_build)
        thread_run16bit.start()
        self.wait_purebit_build()
        self.bw_json_path = os.path.join(self.output_16bit_dir, 'final_operator_internal_bw.json')
        self.bw_json_context = parse_json(self.bw_json_path)
        self.opt_model_path, self.opt_model_name = get_optmodel_path(self.output_16bit_dir)
        self.opt_model = onnx.load_model(self.opt_model_path)
        self.net_struct_json, self.model_initial_name_list = get_connection_net(self.bw_json_context, self.opt_model)
        self.net_out_tensor_names = get_output_tensors(self.net_struct_json)
        self.total_cycle_16bit, self.ave_rmse_16bit = get_total_cycle_mse(self.output_16bit_dir, self.net_out_tensor_names)
        self.ave_cossim_16bit = get_average_cossim(self.output_16bit_dir, self.net_out_tensor_names)
        self.total_cycle_8bit, self.ave_rmse_8bit = get_total_cycle_mse(self.output_8bit_dir, self.net_out_tensor_names)
        self.ave_cossim_8bit = get_average_cossim(self.output_8bit_dir, self.net_out_tensor_names)        
        
    def dump_initbit_result(self):
        print("========================FINISH 8BIT 16BIT INIT RUN====================================")
        print("8  bit cycle:{} ave_rmse:{}, ave_cossim:{}".format(self.total_cycle_8bit,
                                                                '%.8f'%self.ave_rmse_8bit, '%.6f'%self.ave_cossim_8bit))
        print("16 bit cycle:{} ave_rmse:{}, ave_cossim:{}".format(self.total_cycle_16bit,
                                                                '%.8f'%self.ave_rmse_16bit, '%.6f'%self.ave_cossim_16bit))
        print("===============================START MIX QUANT========================================")   
    
    def compute_target_performance(self, cycle_ratio: float, rmse_ratio: float):
        self.target_rmse = (self.ave_rmse_8bit - self.ave_rmse_16bit) * (1. - float(rmse_ratio)) + self.ave_rmse_16bit
        self.target_cycle = int((1. - float(cycle_ratio)) \
            * (self.total_cycle_16bit - self.total_cycle_8bit) + self.total_cycle_8bit)
    
    def set_opt_model_of_onnx(self):
        ini_file = open(self.ini_16bit_path, 'r', encoding='utf-8')
        ini_json = json.load(ini_file)
        out_opt_dir = os.path.join(self.output_base_dir, "opt-model")
        create_dir(out_opt_dir)
        out_opt_file_dir = os.path.join(out_opt_dir, self.opt_model_name)
        copyfile(self.opt_model_path, out_opt_file_dir)
        dst_ini_path = os.path.join(out_opt_dir, get_ini_name(self.ini_16bit_path)+'.ini')
        ini_json['netConfig']['net_path'] = out_opt_file_dir
        if ini_json['netConfig']['framework'] == 'caffe':
            ini_json['netConfig']['framework'] = 'onnx'
            del ini_json['netConfig']['weight_path']
        ini_out_file = open(dst_ini_path, 'w', encoding='utf-8')
        json.dump(ini_json, ini_out_file, ensure_ascii=False, indent=1)
        ini_file.close()
        ini_out_file.close()
        return dst_ini_path
    
    def set_opt_model_mix_ini(self, use_opt=False):
        self.ini_mix_path = self.set_opt_model_of_onnx() if use_opt else self.ini_16bit_path
        logger.info("Use the opt model and store it in '%s'"%self.opt_model_path)
    
    def dump_base_performance(self):
        table_data = {'bit_num':['8bit', '16bit'], 'cycle':['%d'%self.total_cycle_8bit, '%d'%self.total_cycle_16bit],
                      'ave_rmse':['%.8f'%self.ave_rmse_8bit, '%.8f'%self.ave_rmse_16bit],
                      'ave_cossim':['%.6f'%self.ave_cossim_8bit, '%.6f'%self.ave_cossim_16bit]}
        init_df_data = pd.DataFrame(table_data)
        with pd.ExcelWriter(self.record_excel_path, engine='xlsxwriter') as init_df_writer:
            init_df_data.to_excel(init_df_writer, sheet_name=self.sheet_name, startcol=1, index=False) 
    
    def dump_mix_quant_log(self):
        logger.info("===============================================================================")
        logger.info("     /\        | | |  \       /  |          / __ \                        ")
        logger.info("    /  \   _ __| |_|   \     /   |'_'__  __| |  | |_   _  __ _ _ __ _| |_ ")
        logger.info("   / /\ \ | '__| __| |\ \   / /| || |\ \/ /| |  | | | | |/ _` | '_ \|____|")
        logger.info("  / ____ \| |  | |_| | \ \_/ / | || | >  < | |__| | |_| | (_| | | | || |_ ")
        logger.info(" /_/    \_\_|   \__|_|  \___/  |_||_|/_/\_\ \____\_\__,_|\__,_|_| |_| \__|")
        logger.info("======================Copywrite by Artosyn 2023.09.05==========================\n") 
                
    def get_max_length_of_block(self):
        nodes_info = self.blk_final_nodes_info
        block_lens = []
        set_block_dicts = {}
        for block_name in list(nodes_info.keys()):
            block_nodes = nodes_info[block_name]
            set_block_list = []
            for node_name in block_nodes:
                output_tensor_names = block_nodes[node_name]["output_tensor_names"]
                for tensor_name in output_tensor_names:
                    correspond_initials = find_initializer_from_node(tensor_name, self.opt_model, 
                                                                     self.model_initial_name_list)
                    for set_initial_name in correspond_initials:
                        if set_initial_name not in set_block_list:
                            set_block_list.append(set_initial_name)
                insert_node_name = node_name + '_%s'%block_nodes[node_name]["operator_type"]
                if insert_node_name in self.bw_json_context.keys():
                    set_block_list.append(insert_node_name)
            block_lens.append(len(set_block_list))
            set_block_dicts[block_name] = set_block_list
        self.blk_max_length = max(block_lens)
        self.set_blocks_dict = set_block_dicts              
        
class ModuleCodec(object):
    def __init__(self, net_json):
        #super(fastModuleCodec, self)
        self.net_struct_json = net_json
    
    def get_net_output_nodes(self):
        net_dicts = self.net_struct_json
        net_output_nodes = {}
        for op_name in list(net_dicts.keys()):
            _Flag = False
            output_names = net_dicts[op_name]['output_tensor_names']
            for name in list(net_dicts.keys()):
                input_names = net_dicts[name]['input_tensor_names']
                dupname_list = [x for x in output_names if x in input_names]
                if dupname_list:
                    _Flag = True
                    break
            if not _Flag:
                net_output_nodes[op_name] = net_dicts[op_name]
        return net_output_nodes
    
    def find_opname_from_outnames(self, out_name: str):
        re_name = ""
        for op_name in list(self.net_struct_json.keys()):
            cur_output_names = self.net_struct_json[op_name]["output_tensor_names"]
            if out_name in cur_output_names:
                re_name = op_name
                break
        if re_name:
            return re_name
        else:
            logger.error(out_name + " is no corresponding op_name!")
            assert(0)
    
    def find_opname_from_inputnames(self, input_name):
        re_name = []
        for op_name in list(self.net_struct_json.keys()):
            cur_input_names = self.net_struct_json[op_name]["input_tensor_names"]
            if input_name in cur_input_names:
                re_name.append(op_name)
        if re_name:
            return re_name
        else:
            logger.error(input_name + " is no corresponding op_name!")
            assert(0)
    
    def loop_inter_dupgroup(self, cur_output_nodes: dict):
        def inter_duplicate_nodes(input_nodes: dict):
            output_nodes={}
            for _index_, _name_ in enumerate(input_nodes.keys()):
                if _index_ == 0:
                    output_nodes[_name_] = input_nodes[_name_]
                else:
                    repeat_flag=False
                    repeat_names=[]
                    for sec_name in list(output_nodes.keys()):
                        name_in_list = list(output_nodes[sec_name].keys())
                        name_out_list = list(input_nodes[_name_].keys())
                        repeat_name_list = [x for x in name_in_list if x in name_out_list]
                        if repeat_name_list:
                            repeat_names.append(sec_name)
                            repeat_flag = True
                    if repeat_flag:
                        output_nodes_cp = copy.deepcopy(output_nodes)
                        for i, repeat_name in enumerate(repeat_names):
                            if i == 0:
                                continue
                            else:
                                output_nodes[repeat_names[0]].update(output_nodes_cp[repeat_name])
                                del output_nodes[repeat_name]
                        output_nodes[repeat_names[0]].update(input_nodes[_name_])
                    else:
                        output_nodes[_name_] = input_nodes[_name_]
            return output_nodes
        
        while True:
            in_len = len(cur_output_nodes.keys())
            cur_output_nodes = inter_duplicate_nodes(cur_output_nodes)
            out_len = len(cur_output_nodes.keys())
            if in_len == out_len:
                break
        return cur_output_nodes
    
    def integrate_block(self, out_nodes: dict):       
        blk_nodes_info = {}
        net_dicts = self.net_struct_json
        net_dicts_cp = copy.deepcopy(net_dicts)
        for last_out_name in list(out_nodes.keys()):
            del net_dicts_cp[last_out_name]
        output_nodes = copy.deepcopy(out_nodes)
        block_id = 0
        while len(net_dicts_cp) >= 1:
            cur_output_nodes = {}
            for _index, out_name in enumerate(output_nodes.keys()):
                cur_out_node = {}
                out_node = output_nodes[out_name]
                input_of_output_names = out_node['input_tensor_names']
                input_of_output_operator = out_node['operator_type']
                if input_of_output_operator != 'ArtConcat':
                    for in_of_out_name in input_of_output_names:
                        op_name = self.find_opname_from_outnames(in_of_out_name)
                        in_of_out_node = net_dicts[op_name]
                        if op_name not in list(cur_out_node.keys()):
                            cur_out_node[op_name] = in_of_out_node
                    if cur_out_node:
                        cur_output_nodes['OutBlk_%d'%_index] = cur_out_node
                else:
                    for _id, in_of_out_name in enumerate(input_of_output_names):
                        if in_of_out_name not in net_dicts:
                            in_of_out_name = self.find_opname_from_outnames(in_of_out_name)
                        in_of_out_node = net_dicts[in_of_out_name]
                        if in_of_out_name not in cur_out_node.keys():
                            cur_out_node[in_of_out_name] = in_of_out_node
                        if cur_out_node:
                            cur_output_nodes['OutBlk_{}_{}'.format(_index, _id)] = cur_out_node
            cur_output_nodes = self.loop_inter_dupgroup(cur_output_nodes)
            output_nodes = {}
            for _index, cur_out_name in enumerate(cur_output_nodes.keys()):
                sec_nodes = cur_output_nodes[cur_out_name]
                blk_nodes_info['blk_%d'%block_id] = sec_nodes
                block_id += 1
                for sec_node_name in list(sec_nodes.keys()):
                    if sec_node_name in net_dicts_cp:
                        del net_dicts_cp[sec_node_name]
                    output_nodes[sec_node_name] = sec_nodes[sec_node_name]
            blk_nodes_info = self.loop_inter_dupgroup(blk_nodes_info)
        return blk_nodes_info
    
    def get_input_equal_output_with_precision(self):
        net_dicts = self.net_struct_json
        reserve_nodes={}
        id=0
        for op_name in list(net_dicts.keys()):
            operator=net_dicts[op_name]["operator_type"]
            if "Art" not in operator:
                operator = "CallBack"
            if operator in INPUT_OUTPUT_PRECISION_SAME_NODES:
                focus_nodes={}
                focus_nodes[op_name]=net_dicts[op_name]
                input_op_names=net_dicts[op_name]["input_tensor_names"]
                for input_op_name in input_op_names:
                    if input_op_name not in net_dicts.keys():
                        input_op_name=self.find_opname_from_outnames(input_op_name)
                    focus_nodes[input_op_name]=net_dicts[input_op_name] ##可能需要添加本算子
                reserve_nodes["iVoSameBlk_%d" % id]=focus_nodes
                id+=1
        reserve_nodes=self.loop_inter_dupgroup(reserve_nodes)
        return reserve_nodes

    def integrate_normal_special_block(self, normal_blks_info: dict, special_blks_info: dict):
        blk_nodes_info = {}
        out_init_infos = {}
        for normal_blk_name in list(normal_blks_info.keys()):
            normal_blk_nodes = normal_blks_info[normal_blk_name]
            for special_blk_name in list(special_blks_info.keys()):
                special_blk_nodes = special_blks_info[special_blk_name]
                normal_blk_list = list(normal_blk_nodes.keys())
                special_blk_list = list(special_blk_nodes.keys())
                repeat_name_list = [x for x in normal_blk_list if x in special_blk_list]
                if repeat_name_list:
                    normal_blk_nodes.update(special_blk_nodes)
            out_init_infos[normal_blk_name] = normal_blk_nodes
        out_init_infos = self.loop_inter_dupgroup(out_init_infos)
        for idx, blk_name in enumerate(out_init_infos.keys()):
            blk_nodes_info["blk_%d" % idx]= out_init_infos[blk_name]
        return blk_nodes_info
    
    def delete_special_nodes(self, nodes_info: dict, output_nodes: dict):
        net_dicts = self.net_struct_json
        final_nodes_info = {}
        id=0
        for _name_ in list(nodes_info.keys()):
            cur_nodes=nodes_info[_name_]
            del_flag=False
            for _sec_name_ in list(cur_nodes.keys()):
                node=cur_nodes[_sec_name_]
                if _sec_name_ in output_nodes:
                    del_flag = True
                    break
                output_tensor_names = node["output_tensor_names"]
                for output_tensor_name in output_tensor_names:
                    del_op_names = self.find_opname_from_inputnames(output_tensor_name)
                    for del_op_name in del_op_names:
                        del_node = net_dicts[del_op_name]
                        if (del_node["operator_type"] == "ArtScale") and (len(del_node["input_tensor_names"]) >= 2):
                            del_flag = True
                            break
                    if del_flag:
                        break
            if not del_flag:
                final_nodes_info["blk_%d" % id]=cur_nodes
                id+=1
        return final_nodes_info
        
    def get_block_node_infos(self):
        logger.info('Network structure codec 1 / 5')
        net_output_nodes = self.get_net_output_nodes()
        logger.info('Network structure codec 2 / 5')
        normal_blk_nodes_info = self.integrate_block(net_output_nodes)
        logger.info('Network structure codec 3 / 5')
        spec_blk_nodes_info = self.get_input_equal_output_with_precision()
        logger.info('Network structure codec 4 / 5')
        blk_nodes_info = self.integrate_normal_special_block(normal_blk_nodes_info, spec_blk_nodes_info)
        logger.info('Network structure codec 5 / 5')
        blk_final_nodes_info = self.delete_special_nodes(blk_nodes_info, net_output_nodes)
        return blk_final_nodes_info
        
        