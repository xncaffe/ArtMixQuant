import queue
import progressbar
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from progressbar import *
from normal.dge_alg import *

class normalMixRunSearchThread(threading.Thread):
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
                dst_ini_path = task[1]
                self.bar.update(id)
            except:
                break
            try:
                run_artstudio(self.runner.art_studio_excutor_path, 
                              dst_ini_path,
                              self.runner.output_base_dir,
                              self.runner.use_src,
                              self.runner.log_level,
                              self.run_compiler)
            finally:
                self.q.task_done()
    
class NormalProControlEval(MixBaseRuntime):
    def __init__(self, base_params: MixBaseRuntime, args):
        super().__init__(base_params.src_ini_path, base_params.artstudio_root_path, 
                         base_params.output_base_dir, base_params.use_src, base_params.thread_num)
        self.base_params = base_params
        self.best_bw_path = os.path.join(base_params.mix_quant_json_hub, 'bestMixJson')
        self.search_space = self.mix_quant_json_to_search_array()
        self.iter_num = int(args.r)
        self.args = args
        self.penalty = args.PF
        self.iter_rslts = {}
        self.batch_tile_size = 100
        self.reward_loss_curve_dir = os.path.join(base_params.output_base_dir, 'reward_loss_case_curve.png')
        self.cycel_rmse_dot_dir = os.path.join(base_params.output_base_dir, 'iter_cycle_rmse_vary.png')
        create_dir(self.best_bw_path)
        self.COLUMNS_LIST = [chr(x) for x in np.arange(65, 91)]
        
    def mix_quant_json_to_search_array(self):
        net_block_json = self.base_params.blk_final_nodes_info
        precision_array=[]
        for block_name in net_block_json:
            block_infos = net_block_json[block_name]
            _param_=[0,0]
            extFlag=False
            for node_name in block_infos.keys():
                node = block_infos[node_name]
                if(node["output_tensor_bw"] == 8):
                    extFlag=True
                    break 
                _param_[0 if extFlag else 1] = 1             
            precision_array.append(_param_)
        return precision_array
    
    def set_mix_json_by_precision_array(self, set_bit_array):
        set_bw_json = delete_invalid_param(self.base_params.bw_json_context)
        set_op_names_list = []
        for _idx_, block_name in enumerate(self.base_params.blk_final_nodes_info):
            set_bw = set_bit_array[_idx_]
            set_tensor_names_list = self.base_params.set_blocks_dict[block_name]
            for tensor_name in set_tensor_names_list:
                if tensor_name not in set_op_names_list and tensor_name in set_bw_json:
                    set_bw_json[tensor_name]['bit_width'] = set_bw
                    set_op_names_list.append(tensor_name)
        return set_bw_json      
    
    def get_reward_value(self, iter_id):
        def get_linear_score(value, min_val, max_val):
            value = min(value, max_val)
            value = max(value, min_val)
            return (value - min_val) / (max_val - min_val)
        rmse_score = 1 - get_linear_score(self.iter_rslts[iter_id].ave_rmse, 
                                          self.base_params.ave_rmse_16bit, 
                                          self.base_params.ave_rmse_8bit)
        if rmse_score > 0.5:
            rmse_score *= 3
        elif self.penalty:
            rmse_score = rmse_score * (2 if rmse_score > 0.1 else 0.5)
        else:
            rmse_score *= rmse_score
        cycle_score = get_linear_score(self.iter_rslts[iter_id].cycle, 
                                       self.base_params.total_cycle_8bit, 
                                       self.base_params.total_cycle_16bit)
        if cycle_score > 0.9:
            cycle_score *= 10
        elif cycle_score > 0.7:
            cycle_score *= 3
        else:
            cycle_score *= 0.5
        return rmse_score - cycle_score 
    
    def check_best_record_by_iter(self, iter_id:int, best_record_list:list):
        for iter_param in best_record_list:
            if iter_id == iter_param.iter_id:
                return True
        return False
    
    def dump_iter_params_with_record(self, best_record_list: list):
        def style_apply_color(series, record_list):
            colors_list = list()
            for idx, col in enumerate(series):
                if idx in [0, 1]:
                    cur_color = 'background-color: #FF0000; color: #FFFFFF'
                elif idx > 2 and self.check_best_record_by_iter(idx-3, record_list):
                    cur_color = 'background-color: #AFEEEE'
                else:
                    cur_color = ''
                cur_color += '; vertical-align: middle'
                colors_list.append(cur_color)
            return colors_list
        
        header_list = ['case_iter', 'total_cycle', 'ave_rmse', 'ave_cossim', 'reward_loss']
        data_param_list = [['8bit', '16bit', ''],
                           [self.base_params.total_cycle_8bit, self.base_params.total_cycle_16bit, ''],
                           [self.base_params.ave_rmse_8bit, self.base_params.ave_rmse_16bit, ''],
                           [self.base_params.ave_cossim_8bit, self.base_params.ave_cossim_16bit, ''],
                           ['', '', '']]
        for iter_id in self.iter_rslts:
            cur_iter_data_list = ['iter-%d'%iter_id, self.iter_rslts[iter_id].cycle, 
                                  self.iter_rslts[iter_id].ave_rmse, self.iter_rslts[iter_id].ave_cossim,
                                  self.iter_rslts[iter_id].reward_value]
            for _dx_ in range(len(data_param_list)):
                data_param_list[_dx_].append(cur_iter_data_list[_dx_])
        iter_record_data = dict(zip(header_list, data_param_list))
        iter_record_pd_data = pd.DataFrame(iter_record_data)
        cur_writer = pd.ExcelWriter(self.record_excel_path, engine='xlsxwriter')
        style_iter_df_data = iter_record_pd_data.style.apply(style_apply_color, record_list=best_record_list)
        style_iter_df_data.to_excel(cur_writer, sheet_name=self.sheet_name, index=False)
        
        cur_iter_num = len(self.iter_rslts)
        best_start_row = cur_iter_num + 4 * 2
        best_header_list = copy.deepcopy(header_list)
        best_header_list[0] = 'best_iter'
        best_header_list.append('relate_distance')
        best_param_list = [[] for i in range(len(best_header_list))]
        for best_iter_param in best_record_list:
            cur_best_iter_data_list = ['iter-%d'%best_iter_param.iter_id, best_iter_param.cycle,
                                       best_iter_param.ave_rmse, best_iter_param.ave_cossim,
                                       best_iter_param.reward_value, best_iter_param.rel_distance]
            for _dx_ in range(len(cur_best_iter_data_list)):
                best_param_list[_dx_].append(cur_best_iter_data_list[_dx_])
        best_iter_data = dict(zip(best_header_list, best_param_list))
        best_iter_pd_data = pd.DataFrame(best_iter_data)
        style_best_df_data = best_iter_pd_data.style.applymap(lambda v:"background-color: #AFEEEE; vertical-align: middle")
        style_best_df_data.to_excel(cur_writer, sheet_name=self.sheet_name, startrow=best_start_row, index=False)
        
        cur_workbook = cur_writer.book
        cur_worksheet = cur_writer.sheets[self.sheet_name]
        wk_param = cur_workbook.add_format({'bg_color': '#FFFF00', 'bold': True, 'align': 'center', 'border': 2})
        normal_row_param = cur_workbook.add_format({'bottom': 1, 'left': 1, 'right': 1})
        first_iter_param = cur_workbook.add_format({'border': 1})
        fp_num_format_param = cur_workbook.add_format({'num_format': '0.000000'})
        for iter_row in range(iter_record_pd_data.index.size):
            data_range = '{}{}:{}{}'.format(self.COLUMNS_LIST[0], iter_row+2, self.COLUMNS_LIST[4], iter_row+2)
            data_fp_range = '{}{}:{}{}'.format(self.COLUMNS_LIST[2], iter_row+2, self.COLUMNS_LIST[4], iter_row+2)
            if iter_row < 2 or iter_row > 3:
                cur_worksheet.conditional_format(data_range, {'type': 'no_blanks', 'format': normal_row_param})
            elif iter_row == 3:
                cur_worksheet.conditional_format(data_range, {'type': 'no_blanks', 'format': first_iter_param})
            else:
                continue
            cur_worksheet.conditional_format(data_fp_range, {'type': 'no_blanks', 'format': fp_num_format_param})
        iter_header_range = '{}{}:{}{}'.format(self.COLUMNS_LIST[0], 1, self.COLUMNS_LIST[4], 1)
        cur_worksheet.conditional_format(iter_header_range, {'type': 'no_blanks', 'format': wk_param})
        # for iter_col in range(len(header_list)):
        #     cur_worksheet.write(0, iter_col, header_list[iter_col], wk_param)
        best_data_range = '{}{}:{}{}'.format(self.COLUMNS_LIST[0], best_start_row+2, 
                                             self.COLUMNS_LIST[5], best_start_row+2+best_iter_pd_data.index.size)
        best_fp_data_range = '{}{}:{}{}'.format(self.COLUMNS_LIST[2], best_start_row+2, 
                                             self.COLUMNS_LIST[5], best_start_row+2+best_iter_pd_data.index.size)
        cur_worksheet.conditional_format(best_data_range, {'type': 'no_blanks', 'format': normal_row_param})
        cur_worksheet.conditional_format(best_fp_data_range, {'type': 'no_blanks', 'format': fp_num_format_param})
        #for best_col in range(len(best_header_list)):
        best_header_range = '{}{}:{}{}'.format(self.COLUMNS_LIST[0], best_start_row+1, self.COLUMNS_LIST[5], best_start_row+1)
        cur_worksheet.conditional_format(best_header_range, {'type': 'no_blanks', 'format': wk_param})       
        cur_worksheet.set_column('{}:{}'.format(self.COLUMNS_LIST[0], self.COLUMNS_LIST[5]), 10)
        cur_writer.save()
        
    def draw_state_figure(self, best_record_list:list=[], switch_case:str='loss'):
        fig = plt.figure()
        if switch_case == 'distance':
            cycle_array_list = [self.base_params.total_cycle_8bit, self.base_params.total_cycle_16bit]
            sum_rmse_array_list = [self.base_params.ave_rmse_8bit*len(self.base_params.net_out_tensor_names),
                               self.base_params.ave_rmse_16bit*len(self.base_params.net_out_tensor_names)]
            for iter_id in self.iter_rslts:
                if not self.check_best_record_by_iter(iter_id, best_record_list):
                    cycle_array_list.append(self.iter_rslts[iter_id].cycle)
                    sum_rmse_array_list.append(self.iter_rslts[iter_id].ave_rmse*len(self.base_params.net_out_tensor_names))
            for best_iter_param in best_record_list:
                cycle_array_list.append(best_iter_param.cycle)
                sum_rmse_array_list.append(best_iter_param.ave_rmse*len(self.base_params.net_out_tensor_names))
            cycle_array = np.array(cycle_array_list, dtype=np.int64)
            sum_rmse_array = np.array(sum_rmse_array_list, dtype=np.float64)
            plt.plot(cycle_array[:2], sum_rmse_array[:2], 'ro')
            plt.plot(cycle_array[:2], sum_rmse_array[:2], 'r')
            best_iter_size = len(best_record_list)
            common_iter_size = len(self.iter_rslts) - best_iter_size
            if common_iter_size > 0:
                plt.plot(cycle_array[2:(2+common_iter_size)], sum_rmse_array[2:(2+common_iter_size)], 'bo')
            if best_iter_size > 0:
                plt.plot(cycle_array[(2+common_iter_size):], sum_rmse_array[(2+common_iter_size):], 'yo')
            plt.xlabel('output_sum_rmse')
            plt.ylabel('total_cycle')
            plt.grid(True)
            fig.savefig(self.cycel_rmse_dot_dir)
        elif switch_case == 'loss':
            iter_id_array = np.array(list(self.iter_rslts.keys()), dtype=np.int32)
            reward_loss_array = np.array([self.iter_rslts[i].reward_value for i in self.iter_rslts], dtype=np.float32)
            plt.plot(iter_id_array, reward_loss_array, 'b')
            plt.xlabel('case_iter')
            plt.ylabel('reward_loss')
            plt.grid(True)
            fig.savefig(self.reward_loss_curve_dir)
        else:
            logger.error("State figure do not support switch case: '%s'"%switch_case)
            assert(0)
        plt.close(fig)
    
    def dump_cur_tile_size_loss(self, tile_size=100):
        start_iter = len(self.iter_rslts) - tile_size
        cur_tile_iters = list(self.iter_rslts.keys())[start_iter:]
        tile_reward_loss_array = np.array([self.iter_rslts[iter_id].reward_value for iter_id in cur_tile_iters])
        tile_total_reward_loss = np.sum(tile_reward_loss_array)
        tile_max_id = np.argmax(tile_reward_loss_array)
        tile_min_id = np.argmin(tile_reward_loss_array)
        tile_mean_reward_loss = tile_total_reward_loss / tile_size
        logger.info("The iteration index: %d (best effect in current batch) || "%cur_tile_iters[tile_max_id] 
                    + "corresponding reward loss = %.6f"%self.iter_rslts[cur_tile_iters[tile_max_id]].reward_value)
        logger.info("The iteration index: %d (worst performance in current batch) || "%cur_tile_iters[tile_min_id] 
                    + "corresponding reward loss = %.6f"%self.iter_rslts[cur_tile_iters[tile_min_id]].reward_value)
        logger.info("The current batch average reward loss = %.6f."%tile_mean_reward_loss)
    
    def dump_top_optimal_mix_result(self, best_record_list):
        logger.info("Normal mode mix quant search has been completed, the following is the top performance:")
        logger.info("################################################################################################################")
        for best_record_param in best_record_list:
            logger.info("iterate index:%7d || "%best_record_param.iter_id 
                        + "total cycle:%15d || "%best_record_param.cycle 
                        + "ave rmse:%10.8f || "%best_record_param.ave_rmse
                        + "ave cossim:%7.6f || "%best_record_param.ave_cossim
                        + "relate distance:%15.6f"%best_record_param.rel_distance)
        logger.info("################################################################################################################")
        
    def run_normal_iterate(self):
        remain_iter = self.iter_num % self.batch_tile_size
        batch_num = self.iter_num // self.batch_tile_size + (1 if remain_iter > 0 else 0)
        logger.info("Start normal mode mixed precision quantization iterative search, iterating ...")
        dge_alg_runner = DiscreteGeneticEvolutionaryAlg(self.args, self.search_space, 
                                                        self.base_params, self.best_bw_path)
        for batch_id in range(batch_num):
            cur_batch_size = remain_iter if (batch_id == batch_num - 1) and remain_iter > 0 else self.batch_tile_size
            remain_batch_size = cur_batch_size % self.thread_num
            work_tile_num = cur_batch_size// self.thread_num + (1 if remain_batch_size != 0 else 0)
            start_tile_iter = batch_id * self.batch_tile_size
            end_tile_iter = start_tile_iter + cur_batch_size
            widgets = ['iter %d => '%start_tile_iter + '%d Progress: '%end_tile_iter, Percentage(), ' ', Bar('#'), ' ', Timer()]
            bar = progressbar.ProgressBar(widgets=widgets, maxval=cur_batch_size).start()
            for tile_work_id in range(work_tile_num):
                run_batch_work_dict = {}
                run_queue = queue.Queue()
                for thread_id in range(self.thread_num):
                    cur_batch_id = tile_work_id * self.thread_num + thread_id
                    if cur_batch_id >= cur_batch_size:
                        break
                    cur_iter_id = start_tile_iter + cur_batch_id
                    cur_bit_param = dge_alg_runner.get_new_param()
                    set_search_precision = [[8, 16][cur_bit_option] for cur_bit_option in cur_bit_param]
                    dst_bw_json_context = self.set_mix_json_by_precision_array(set_search_precision)
                    dst_bw_json_path = os.path.join(self.mix_quant_json_hub, get_ini_name(self.base_params.ini_mix_path)) \
                        + '_bw_%d.json'%cur_iter_id
                    save_json(dst_bw_json_context, dst_bw_json_path)
                    dst_ini_path = self.base_params.ini_mix_path[:-4] + '-worker-%d.ini'%thread_id
                    set_ini_bw_json(self.base_params.ini_mix_path, 
                                dst_ini_path, dst_bw_json_path, -1)
                    run_queue.put([cur_batch_id, dst_ini_path])
                    run_batch_work_dict[thread_id] = [dst_ini_path, cur_iter_id]
                    self.iter_rslts[cur_iter_id] = SimpleNamespace(cycle=0, ave_rmse=0., ave_cossim=0., 
                                                                   rel_distance=0., reward_value=0., 
                                                                   iter_id=cur_iter_id, current_param=cur_bit_param,
                                                                   mix_bw_json_path=dst_bw_json_path,
                                                                   best_mix_bw_json_path=None)
                for i in range(self.thread_num):
                    normal_iter_worker = normalMixRunSearchThread(run_queue, self, bar, run_compiler=True)
                    normal_iter_worker.start()
                run_queue.join()
                for worker_id in run_batch_work_dict.keys():
                    iter_id = run_batch_work_dict[worker_id][1]
                    iter_ini_path = run_batch_work_dict[worker_id][0]
                    cur_out_dir = os.path.join(self.output_base_dir, get_ini_name(iter_ini_path))
                    _cycle_, _rmse_ = get_total_cycle_mse(output_dir=cur_out_dir,
                                                        tensor_name=self.base_params.net_out_tensor_names)
                    _cossim_ = get_average_cossim(output_dir=cur_out_dir,
                                                tensor_name=self.base_params.net_out_tensor_names)
                    self.iter_rslts[iter_id].cycle = _cycle_
                    self.iter_rslts[iter_id].ave_rmse = _rmse_
                    self.iter_rslts[iter_id].ave_cossim = _cossim_
                    self.iter_rslts[iter_id].reward_value = self.get_reward_value(iter_id)
                    if self.iter_rslts[iter_id].reward_value != 0:
                        related_cycle = float(_cycle_) / float(self.base_params.total_cycle_16bit)
                        related_ave_rmse = _rmse_ / self.base_params.ave_rmse_8bit
                        self.iter_rslts[iter_id].rel_distance = np.sqrt(np.square(related_cycle) + np.square(related_ave_rmse))
                    dge_alg_runner.update_reward(self.iter_rslts[iter_id])
                    if os.path.exists(iter_ini_path):
                        os.system("rm -r {}".format(iter_ini_path))
                self.dump_iter_params_with_record(dge_alg_runner.best_record)
                self.draw_state_figure(dge_alg_runner.best_record, 'distance')
                self.draw_state_figure()
            bar.finish()
            self.dump_cur_tile_size_loss(cur_batch_size)
        self.dump_top_optimal_mix_result(dge_alg_runner.best_record)
        
def normal_mix_run(mix_params, args):
    normal_runner = NormalProControlEval(mix_params, args)
    normal_runner.run_normal_iterate()
    logger.info("Mix Quant Precision Finish With Normal Mode!")
    
        