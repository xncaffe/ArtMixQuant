import queue
import progressbar
import numpy as np
from progressbar import *
from types import SimpleNamespace
from common.mixQuantCodec import *

class fastMixRunSearchThread(threading.Thread):
    def __init__(self, q, runner, bar, run_compiler=False):
        threading.Thread.__init__(self)
        self.bar = bar
        self.q = q
        self.runner = runner
        self.run_compiler = run_compiler

    def run(self):
        while True:
            try:
                task = self.q.get(block=True, timeout=1)
                id = task[0]
                dst_bw_json_path = task[1]
                dst_ini_path = task[2]
                self.bar.update(id)
            except:
                break
            try:
                bw_json = task[3].per_bw_json
                save_json(bw_json, dst_bw_json_path)
                set_ini_bw_json(self.runner.base_params.ini_mix_path, 
                                dst_ini_path, dst_bw_json_path, -1)
                run_artstudio(self.runner.art_studio_excutor_path, 
                              dst_ini_path,
                              self.runner.output_base_dir,
                              self.runner.use_src,
                              self.runner.log_level,
                              self.run_compiler)
            finally:
                self.q.task_done()

class FastProControlEval(MixBaseRuntime):
    MAX_BLOCK_ID = 100000
    def __init__(self, base_params: MixBaseRuntime):
        super().__init__(base_params.src_ini_path, base_params.artstudio_root_path, 
                         base_params.output_base_dir, base_params.use_src, base_params.thread_num)
        self.base_params = base_params
        self.ini_valid_path = base_params.ini_mix_path[:-4]+'_fastprocessing.ini'
        self.dst_final_json_dir = os.path.join(self.mix_quant_json_hub, 
                                               get_ini_name(base_params.ini_mix_path)+'_fast_rslt.json')
        self.fast_mix_per_quant_params = {}
        self.sorted_per_block_names_list = []
        self.fast_mix_stack_quant_params = {}
        self.re_block_name = None
        self.target_block_name = None
        self.target_stack_id = FastProControlEval.MAX_BLOCK_ID
        self.COLUMNS_LIST = [chr(x) for x in np.arange(65, 91)]
        self.expand_columns_list()
    
    def expand_columns_list(self):
        src_columns_list = copy.deepcopy(self.COLUMNS_LIST)
        for column in src_columns_list[:10]:
            for src_column in src_columns_list:
                self.COLUMNS_LIST.append('{}{}'.format(column, src_column))
        
    def set_fast_mix_bw_json(self, set_bw_json, block_info_dict):
        out_set_json=copy.deepcopy(set_bw_json)
        set_op_name_lists = []
        for block_name in block_info_dict.keys():
            block_nodes = block_info_dict[block_name]
            for op_name in block_nodes.keys():
                node = block_nodes[op_name]
                insert_op_name = op_name + '_{}'.format(node["operator_type"])
                if insert_op_name in out_set_json.keys():
                    set_op_name_lists.append(insert_op_name)
                output_tensor_names = node["output_tensor_names"]
                for output_tensor_name in output_tensor_names:
                    set_initial_lists = find_initializer_from_node(output_tensor_name, 
                                                                   self.base_params.opt_model,
                                                                   self.base_params.model_initial_name_list)
                    for set_initial_name in set_initial_lists:
                        if set_initial_name not in set_op_name_lists:
                            set_op_name_lists.append(set_initial_name)
        for set_op_name in out_set_json.keys():
            if set_op_name in set_op_name_lists:
                out_set_json[set_op_name]["bit_width"] = 8
                out_set_json_op_cp = copy.deepcopy(out_set_json[set_op_name])
                for param_name in out_set_json_op_cp.keys():
                    if param_name == "bit_width":
                        continue
                    del out_set_json[set_op_name][param_name]
            else:
                out_set_json_op_cp = copy.deepcopy(out_set_json[set_op_name])
                for param_name in out_set_json_op_cp.keys():
                    if param_name != "bit_width":
                        del out_set_json[set_op_name][param_name]
        return out_set_json
    
    def loop_run_per_block(self):
        loop_block_names_list = list(self.base_params.blk_final_nodes_info.keys())
        config_len = len(loop_block_names_list)
        remain_len = config_len - (config_len // self.thread_num) * self.thread_num
        worker_tile_num = config_len // self.thread_num \
            + (remain_len if remain_len != 0 else 0)
        logger.info("Starts the first-stage search in fast mode with automatic mixed precision ...")
        widgets = ['Run Per Block Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(loop_block_names_list)).start()
        for w in range(worker_tile_num):
            run_block_names_dict = {}
            run_queue = queue.Queue()
            for t in range(self.thread_num):
                block_id = w * self.thread_num + t
                if block_id >= config_len:
                    break
                block_name = loop_block_names_list[block_id]
                cur_block_info = self.base_params.blk_final_nodes_info[block_name]
                cur_block_param = SimpleNamespace(block_name=block_name,block_info=cur_block_info)
                set_blocks_info_dict = {block_name: cur_block_info}
                cur_block_param.per_bw_json = self.set_fast_mix_bw_json(self.base_params.bw_json_context, 
                                                                        set_blocks_info_dict)
                cur_block_param.cycle = 0
                cur_block_param.ave_rmse = 0.
                cur_block_param.ave_cossim = 0.
                dst_bw_json_name = get_ini_name(self.base_params.ini_mix_path)+'_per_block_thread_%d.json'%t \
                    if self.thread_num > 1 else get_ini_name(self.base_params.ini_mix_path)+'_per_block.json'
                dst_bw_json_path = os.path.join(self.mix_quant_json_hub, dst_bw_json_name)
                dst_mix_ini_path = self.ini_valid_path[:-4] + '_thread_%d.ini'%t if self.thread_num > 1 else self.ini_valid_path
                run_queue.put([block_id, dst_bw_json_path, dst_mix_ini_path, cur_block_param])
                self.fast_mix_per_quant_params[block_name] = cur_block_param
                run_block_names_dict[block_name] = dst_mix_ini_path
            for i in range(self.thread_num):
                fast_per_worker = fastMixRunSearchThread(run_queue, self, bar)
                fast_per_worker.start()
            run_queue.join()
            for run_block_name in run_block_names_dict.keys():
                cur_out_dir = os.path.join(self.output_base_dir, get_ini_name(run_block_names_dict[run_block_name]))
                _cycle_, _rmse_ = get_total_cycle_mse(output_dir=cur_out_dir,
                                                      tensor_name=self.base_params.net_out_tensor_names, 
                                                      dump_cycle=False)
                _cossim_ = get_average_cossim(output_dir=cur_out_dir,
                                              tensor_name=self.base_params.net_out_tensor_names)
                self.fast_mix_per_quant_params[run_block_name].cycle = _cycle_
                self.fast_mix_per_quant_params[run_block_name].ave_rmse = _rmse_
                self.fast_mix_per_quant_params[run_block_name].ave_cossim = _cossim_
                if os.path.exists(run_block_names_dict[run_block_name]):
                    os.system("rm -r {}".format(run_block_names_dict[run_block_name]))
        bar.finish()
        logger.info("First-stage search finish!")
        
    def per_block_qsort_rmse(self):
        self.sorted_per_block_names_list = sorted(self.fast_mix_per_quant_params,
                                                key=lambda x:self.fast_mix_per_quant_params[x].ave_rmse,
                                                reverse=False)
        
    def dump_stack_data_to_record(self, spec_id=MAX_BLOCK_ID):
        first_col_color = '#D3D3D3'
        use_8_row_color = '#AFEEEE'
        spec_row_color = '#FF0000'
        spec_font_color = '#FFFFFF'
        use_16_row_color = '#F0FFFF'
        def style_apply_color(series, spec_id):
            series_name = series.name
            colors_list = list()
            for idx, col in enumerate(series):
                if series_name == 'node_priority_id' and idx != spec_id:
                    cur_color = 'background-color: '+first_col_color
                # elif col == '' and idx != spec_id:
                #     cur_color = ''
                elif idx < spec_id:
                    cur_color = 'background-color: '+use_8_row_color
                elif idx == spec_id:
                    cur_color = 'background-color: {}; color: {}; font-weight: bold'.format(spec_row_color, spec_font_color)
                else:
                    cur_color = 'background-color: '+use_16_row_color
                if col != '':
                    cur_color += '; vertical-align: middle'
                colors_list.append(cur_color)
            return colors_list
        
        src_record_data = pd.read_excel(self.record_excel_path, sheet_name=self.sheet_name)
        src_df_data = pd.DataFrame(src_record_data)
        base_df_data = src_df_data.iloc[:2, :4]
        node_header_list = ['tensor_%d'%i for i in range(self.base_params.blk_max_length)]
        header_list = ['node_priority_id', 'cycle', 'ave_rmse', 'ave_cossim', 'use_8', 'block_name'] \
            + node_header_list
        stack_record_data = {}
        for header_name in header_list:
            stack_record_data[header_name] = []
        for idx, block_name in enumerate(self.sorted_per_block_names_list):
            if block_name in self.fast_mix_stack_quant_params.keys():
                stack_record_data['node_priority_id'].append(idx)
                stack_record_data['block_name'].append(block_name)
                cur_tensor_names_list = self.base_params.set_blocks_dict[block_name]
                for i in range(self.base_params.blk_max_length):
                    stack_record_data['tensor_%d'%i].append(cur_tensor_names_list[i] \
                        if len(cur_tensor_names_list) > i else '')
                stack_record_data['use_8'].append('Yes' if idx <= spec_id else 'No')
                stack_record_data['cycle'].append(self.fast_mix_stack_quant_params[block_name].cycle)
                stack_record_data['ave_rmse'].append(self.fast_mix_stack_quant_params[block_name].ave_rmse)
                stack_record_data['ave_cossim'].append(self.fast_mix_stack_quant_params[block_name].ave_cossim)
            else:
                break
        cur_writer = pd.ExcelWriter(self.record_excel_path, engine='xlsxwriter')
        style_base_df_data = base_df_data.style.applymap(lambda v:"background-color: #99FFCC; vertical-align: middle")
        style_base_df_data.to_excel(cur_writer, sheet_name=self.sheet_name, startcol=0, index=False)
        if spec_id < FastProControlEval.MAX_BLOCK_ID:
            spec_record_data = self.fast_mix_stack_quant_params[self.sorted_per_block_names_list[spec_id]]
            spec_pd_data = pd.DataFrame(dict(zip(list(base_df_data.columns), [['mix'], 
                                [spec_record_data.cycle], [spec_record_data.ave_rmse], 
                                [spec_record_data.ave_cossim]])))
            base_df_data = base_df_data.append(spec_pd_data, ignore_index=True)
            style_base_df_data = base_df_data.style.applymap(lambda v:"background-color: #99FFCC; vertical-align: middle")
            style_base_df_data.to_excel(cur_writer, sheet_name=self.sheet_name, startcol=0, index=False)
        mix_data_start_row = 7
        stack_df_data = pd.DataFrame(stack_record_data)
        style_stack_df_data = stack_df_data.style.apply(style_apply_color, spec_id=spec_id)
        style_stack_df_data.to_excel(cur_writer, sheet_name=self.sheet_name, startrow=mix_data_start_row, index=False)
        
        cur_workbook = cur_writer.book
        cur_worksheet = cur_writer.sheets[self.sheet_name]
        base_header_list = list(base_df_data.columns)
        wk_param = cur_workbook.add_format({'bg_color': '#FFFF00', 'bold': True, 'align': 'center', 'border': 2})
        base_row_param = cur_workbook.add_format({'bg_color': '#99FFCC', 'bold': False, 
                                                  'align': 'center', 'bottom': 1, 'left': 1, 'right': 1})
        first_col_param = cur_workbook.add_format({'bg_color': first_col_color, 'bold': False, 
                                                   'align': 'center', 'bottom': 1, 'left': 1, 'right': 1})
        use_8_row_param = cur_workbook.add_format({'bg_color': use_8_row_color, 'bold': False, 
                                                   'align': 'center', 'bottom': 1, 'left': 1, 'right': 1})
        spec_row_param = cur_workbook.add_format({'bg_color': spec_row_color, 'font_color': spec_font_color, 
                                                  'bold': True, 'align': 'center', 'bottom': 1, 'left': 1, 'right': 1})
        use_16_row_param = cur_workbook.add_format({'bg_color': use_16_row_color, 'bold': False, 
                                                    'align': 'center', 'bottom': 1, 'left': 1, 'right': 1})
        use_8_tensor_param = cur_workbook.add_format({'bg_color': use_8_row_color, 'bold': False, 
                                                      'align': 'center', 'bottom': 1})
        spec_tensor_param = cur_workbook.add_format({'bg_color': spec_row_color, 'font_color': spec_font_color, 
                                                     'bold': False, 'align': 'center', 'bottom': 1})
        use_16_tensor_param = cur_workbook.add_format({'bg_color': use_16_row_color, 'bold': False, 
                                                       'align': 'center', 'bottom': 1})
        use_8_last_param = cur_workbook.add_format({'bg_color': use_8_row_color, 'bold': False, 
                                                    'align': 'center', 'bottom': 1, 'right': 1})
        spec_last_param = cur_workbook.add_format({'bg_color': spec_row_color, 'font_color': spec_font_color, 
                                                   'bold': False, 'align': 'center', 'bottom': 1, 'right': 1})
        use_16_last_param = cur_workbook.add_format({'bg_color': use_16_row_color, 'bold': False, 
                                                     'align': 'center', 'bottom': 1, 'right': 1})
        for base_col in range(len(base_header_list)):
            cur_worksheet.write(0, base_col, base_header_list[base_col], wk_param)
            for base_row in range(base_df_data.index.size):
                cur_worksheet.write(base_row+1, base_col, base_df_data.iloc[base_row, base_col], base_row_param)
        stack_header_list = list(stack_df_data.columns)
        for stack_col in range(len(stack_header_list)):
            cur_worksheet.write(mix_data_start_row, stack_col, stack_header_list[stack_col], wk_param)
        no_merge_end_len = len(header_list) - self.base_params.blk_max_length
        for info_col in range(len(header_list)):
            for info_row in range(stack_df_data.index.size):
                if info_row < spec_id:
                    if info_col == 0:
                        current_param = first_col_param
                    elif info_col < no_merge_end_len:
                        current_param = use_8_row_param
                    elif info_col == len(header_list) - 1:
                        current_param = use_8_last_param
                    else:
                        current_param = use_8_tensor_param
                elif info_row == spec_id:
                    if info_col < no_merge_end_len:
                        current_param = spec_row_param
                    elif info_col == len(header_list) - 1:
                        current_param = spec_last_param
                    else:
                        current_param = spec_tensor_param
                else:
                    if info_col == 0:
                        current_param = first_col_param
                    elif info_col < no_merge_end_len:
                        current_param = use_16_row_param
                    elif info_col == len(header_list) - 1:
                        current_param = use_16_last_param
                    else:
                        current_param = use_16_tensor_param
                cur_worksheet.write(mix_data_start_row+1+info_row, info_col, stack_df_data.iloc[info_row, info_col], current_param)
        performance_cell_range = '{}{}:{}{}'.format(self.COLUMNS_LIST[2], mix_data_start_row+1, 
                                                    self.COLUMNS_LIST[3], mix_data_start_row+1+stack_df_data.index.size)
        fp_num_format_param = cur_workbook.add_format({'num_format': '0.000000'})
        cur_worksheet.set_column('{}:{}'.format(self.COLUMNS_LIST[0], self.COLUMNS_LIST[0]), 20)
        cur_worksheet.set_column('{}:{}'.format(self.COLUMNS_LIST[1], self.COLUMNS_LIST[5]), 12)
        cur_worksheet.conditional_format(performance_cell_range, {'type': 'no_blanks', 'format': fp_num_format_param})         
        stack_header_merge_range = self.COLUMNS_LIST[6:(6 + self.base_params.blk_max_length)]
        merge_range_str = '{}{}:{}{}'.format(stack_header_merge_range[0], mix_data_start_row+1, stack_header_merge_range[-1], mix_data_start_row+1)
        cur_worksheet.merge_range(merge_range_str, 'tensor_names', wk_param)
        cur_writer.save()
        
    def check_target_achieve(self, cur_stack_list):
        for cur_idx, cur_block_name in enumerate(cur_stack_list):
            cur_mix_quant_param = self.fast_mix_stack_quant_params[cur_block_name]
            if cur_mix_quant_param.cycle <= self.base_params.target_cycle \
                and cur_mix_quant_param.ave_rmse <= self.base_params.target_rmse:
                    return cur_idx
        return FastProControlEval.MAX_BLOCK_ID
    
    def get_target_block_name_with_rmse_without_cycle(self):
        logger.warning("############################################################################################")
        logger.warning("The fast method cannot achieve rmse and cycle to meet the set requirements at the same time!")
        logger.warning("Will match the lowest cycle hybrid method that first satisfies the target rmse!")
        logger.warning("############################################################################################")
        reverse_sorted_per_block_id_list = list(range(len(self.sorted_per_block_names_list)))
        reverse_sorted_per_block_id_list.reverse()
        for reverse_block_id in reverse_sorted_per_block_id_list:
            cur_block_name = self.sorted_per_block_names_list[reverse_block_id]
            cur_mix_quant_param = self.fast_mix_stack_quant_params[cur_block_name]
            if cur_mix_quant_param.ave_rmse < self.base_params.target_rmse:
                self.target_block_name = cur_block_name
                self.dump_stack_data_to_record(reverse_block_id)
                break
        if self.target_block_name is None:
            logger.error("No targets found, please check!")
            assert(0)
            
    def dump_fast_mix_search_result(self):
        target_mix_param = self.fast_mix_stack_quant_params[self.target_block_name]
        logger.info("********************************* Fast mixed precision quantization results **********************************")
        logger.info("Only  16 bit -> ave rmse: %.8f || "%self.base_params.ave_rmse_16bit \
            + "sum cycles: %d || "%self.base_params.total_cycle_16bit + "ave cossim: %.6f"%self.base_params.ave_cossim_16bit)
        logger.info("Only   8 bit -> ave rmse: %.8f || "%self.base_params.ave_rmse_8bit \
            + "sum cycles: %d || "%self.base_params.total_cycle_8bit + "ave_cossim: %.6f"%self.base_params.ave_cossim_8bit)
        logger.info("Target   inf -> ave rmse: %.8f || "%self.base_params.target_rmse + "sum cycles: %d || "%self.base_params.target_cycle)
        logger.info("Mix    Quant -> ave rmse: %.8f || "%target_mix_param.ave_rmse + "sum cycles: %d || "%target_mix_param.cycle \
            + "ave_cossim: %.6f"%target_mix_param.ave_cossim)
        logger.info("**************************************************************************************************************")
        
    def stack_run_each_block_quant(self):
        config_len = len(self.sorted_per_block_names_list)
        remain_len = config_len - (config_len // self.thread_num) * self.thread_num
        worker_tile_num = config_len // self.thread_num \
            + (remain_len if remain_len != 0 else 0)
        init_bw_json = copy.deepcopy(self.base_params.bw_json_context)
        logger.info("Starts the second-stage search in fast mode with automatic mixed precision ...")
        widgets = ['Run Mix Quant Search Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=config_len).start()
        set_stack_blocks_info_dict = {}
        for tile_id in range(worker_tile_num):
            run_queue = queue.Queue()
            set_stack_block_names_ini_dict = {}
            for thread_id in range(self.thread_num):
                block_name_index = tile_id * self.thread_num + thread_id
                if block_name_index >= config_len:
                    break
                block_name = self.sorted_per_block_names_list[block_name_index]
                cur_block_info = self.base_params.blk_final_nodes_info[block_name]
                cur_block_param = SimpleNamespace(block_name=block_name, block_info=cur_block_info)
                set_stack_blocks_info_dict[block_name] = cur_block_info
                cur_block_param.per_bw_json = self.set_fast_mix_bw_json(init_bw_json, 
                                                                        set_stack_blocks_info_dict)
                cur_block_param.cycle = 0
                cur_block_param.ave_rmse = 0.
                cur_block_param.ave_cossim = 0.
                dst_bw_json_name = get_ini_name(self.base_params.ini_mix_path)+'_stack_block_%d.json'%block_name_index
                dst_bw_json_path = os.path.join(self.mix_quant_json_hub, dst_bw_json_name)
                dst_mix_ini_path = self.ini_valid_path[:-4] + '_thread_%d.ini'%thread_id if self.thread_num > 1 else self.ini_valid_path
                set_stack_block_names_ini_dict[block_name] = dst_mix_ini_path
                run_queue.put([block_name_index, dst_bw_json_path, dst_mix_ini_path, cur_block_param])
                self.fast_mix_stack_quant_params[block_name] = cur_block_param
            for thread_id in range(self.thread_num):
                fast_stack_worker = fastMixRunSearchThread(run_queue, self, bar, run_compiler=True)
                fast_stack_worker.start()
            run_queue.join()
            for run_block_name in set_stack_block_names_ini_dict.keys():
                cur_out_dir = os.path.join(self.output_base_dir, get_ini_name(set_stack_block_names_ini_dict[run_block_name]))
                _cycle_, _rmse_ = get_total_cycle_mse(output_dir=cur_out_dir,
                                                      tensor_name=self.base_params.net_out_tensor_names, 
                                                      dump_cycle=True)
                _cossim_ = get_average_cossim(output_dir=cur_out_dir,
                                              tensor_name=self.base_params.net_out_tensor_names)
                self.fast_mix_stack_quant_params[run_block_name].cycle = _cycle_
                self.fast_mix_stack_quant_params[run_block_name].ave_rmse = _rmse_
                self.fast_mix_stack_quant_params[run_block_name].ave_cossim = _cossim_
                if os.path.exists(set_stack_block_names_ini_dict[run_block_name]):
                    os.system("rm -r {}".format(set_stack_block_names_ini_dict[run_block_name]))
            target_cur_idx = self.check_target_achieve(list(set_stack_block_names_ini_dict.keys()))
            if target_cur_idx < FastProControlEval.MAX_BLOCK_ID:
                self.target_block_name = list(set_stack_block_names_ini_dict.keys())[target_cur_idx]
                save_json(self.fast_mix_stack_quant_params[self.target_block_name].per_bw_json, self.dst_final_json_dir)
                self.target_stack_id = target_cur_idx + tile_id * self.thread_num
            self.dump_stack_data_to_record(self.target_stack_id)
        bar.finish()
    
    def fast_mix_search_post_process(self):
        if self.target_block_name is None:
            self.get_target_block_name_with_rmse_without_cycle()
        remove_dir(self.ini_valid_path)
        self.dump_fast_mix_search_result()
        logger.info("Mix Quant Precision Finish With Fast Mode!")

def fast_mix_run(mix_params):
    fast_runner = FastProControlEval(mix_params)
    fast_runner.loop_run_per_block()
    fast_runner.per_block_qsort_rmse()
    fast_runner.stack_run_each_block_quant()
    fast_runner.fast_mix_search_post_process()

    
    
        
        
    