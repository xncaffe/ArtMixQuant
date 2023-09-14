import random
import copy
import numpy as np
from types import SimpleNamespace
from common.mixQuantCodec import *

class DiscreteGeneticEvolutionaryAlg(object):
    def __init__(self, args, SEARCH_SPACE, base_params: MixBaseRuntime, best_mix_bw_path:str):
        self.population_size = args.m
        self.random_best_update_filter_probility = args.mp
        self.cross_over_update_filter_probility = args.xovr
        self.SEARCH_SPACE = SEARCH_SPACE
        self.penalty = args.PF
        # self.warmup_flg = False
        # self.generation_counter = 0
        self.current_generations = []
        self.sample_generations_regist = []
        self.new_generations = []
        self.new_generations_regist = []
        self.reward_dict = {}
        self.random_bais = args.rd
        self.init_genertation()
        self.bigger_is_better = True
        self.generation_best_record = []
        self.loss_record = []
        self.remember_best_num = 10
        self.best_record = []
        self.base_params = base_params
        self.best_mix_bw_path = best_mix_bw_path
        self.iter_num = args.r
    
    def init_genertation(self):
        for i in range(self.population_size):
            current_population_single_generate = self.random_generate()
            self.current_generations.append(current_population_single_generate)
            self.sample_generations_regist.append(current_population_single_generate)
            next_population_single_generate = self.random_generate()
            self.new_generations.append(next_population_single_generate)
            self.sample_generations_regist.append(next_population_single_generate)
    
    def get_new_param(self):
        if (len(self.sample_generations_regist) > 0):
            ret_param = self.sample_generations_regist[0]
            self.sample_generations_regist = self.sample_generations_regist[1:]
        else:
            self.current_generations, self.reward_dict = self.selection(self.new_generations, self.current_generations)
            self.record_best()
            self.new_generations = self.gen_new_generation(self.current_generations)
            self.sample_generations_regist = self.new_generations.copy()
            ret_param = self.sample_generations_regist[0]
            self.sample_generations_regist = self.sample_generations_regist[1:]
        return ret_param
    
    def record_best(self):
        best_value = max(self.reward_dict.values()) if self.bigger_is_better else min(self.reward_dict.values())
        for key, value in self.reward_dict.items():
            if (value == best_value):
                self.generation_best_record.append([key, value])
                self.loss_record.append(value)
    
    def reward_dict_key_to_list(self):
        return [list(key) for key in self.reward_dict]
    
    def update_reward(self, iter_params: SimpleNamespace):
        self.reward_dict.setdefault(tuple(iter_params.current_param), iter_params.reward_value)
        if iter_params.ave_rmse <= self.base_params.ave_rmse_8bit:
            best_bw_json_dir = os.path.join(self.best_mix_bw_path, get_ini_name(self.base_params.ini_mix_path)) \
                + '_bw_%d.json'%iter_params.iter_id
            if len(self.best_record) < self.remember_best_num:
                iter_params.best_mix_bw_json_path = best_bw_json_dir
                self.best_record.append(iter_params)
                copyfile(iter_params.mix_bw_json_path, best_bw_json_dir)
                self.best_record = sorted(self.best_record, key=lambda x:x.rel_distance)
            else:
                best_record_len = len(self.best_record)
                best_same_distance_idx_list = []
                for _idx_ in range(best_record_len):
                    cur_record_params = self.best_record[_idx_]
                    if iter_params.rel_distance == cur_record_params.rel_distance \
                        and iter_params.cycle == cur_record_params.cycle:
                            best_same_distance_idx_list.append(_idx_)
                minest_iter_same_distance_id, minest_iter = [self.iter_num+1, self.iter_num]
                for best_same_idx in best_same_distance_idx_list:
                    minest_iter_same_distance_id, minest_iter = [best_same_idx, self.best_record[best_same_idx]] \
                        if self.best_record[best_same_idx].iter_id < minest_iter \
                            else [minest_iter_same_distance_id, minest_iter]
                iter_params.best_mix_bw_json_path = best_bw_json_dir
                if minest_iter_same_distance_id != self.iter_num + 1:
                    if os.path.exists(self.best_record[minest_iter_same_distance_id].best_mix_bw_json_path):
                        os.remove(self.best_record[minest_iter_same_distance_id].best_mix_bw_json_path)
                    copyfile(iter_params.mix_bw_json_path, iter_params.best_mix_bw_json_path)
                    self.best_record[minest_iter_same_distance_id] = iter_params
                    self.best_record = sorted(self.best_record, key=lambda x:x.reward_value, reverse=False)
                else:
                    self.best_record.append(iter_params)
                self.best_record = sorted(self.best_record, key=lambda x:x.rel_distance)
                if len(self.best_record) > best_record_len:
                    if self.best_record[-1].best_mix_bw_json_path is None:
                        logger.error("When updating reward, the optimal configuration bw_json file is None, please check.")
                        assert(0)
                    if self.best_record[-1].iter_id != iter_params.iter_id:
                        copyfile(iter_params.mix_bw_json_path, iter_params.best_mix_bw_json_path)
                    if os.path.exists(self.best_record[-1].best_mix_bw_json_path):
                        os.remove(self.best_record[-1].best_mix_bw_json_path)
                    del self.best_record[-1]  
            
    def gen_new_generation(self, pend_upgrade_list):
        update_list = self.mutation(pend_upgrade_list)
        return self.crossover(pend_upgrade_list, update_list)
    
    def selection(self, cur_gen_list, pre_gen_list):
        new_reward_dict = {}
        for i in range(0, self.population_size):
            key_cur = tuple(cur_gen_list[i])
            reward_cur = self.reward_dict[key_cur]
            key_pre = tuple(pre_gen_list[i])
            reward_pre = self.reward_dict[key_pre]
            if self.bigger_is_better:
                if reward_pre <= reward_cur:
                    pre_gen_list[i] = cur_gen_list[i]
                    new_reward_dict.setdefault(tuple(key_cur), reward_cur)
                else:
                    new_reward_dict.setdefault(tuple(key_pre), reward_pre)
            else:
                if reward_pre >= reward_cur:
                    pre_gen_list[i] = cur_gen_list[i]
                    new_reward_dict.setdefault(tuple(key_cur), reward_cur)
                else:
                    new_reward_dict.setdefault(tuple(key_pre), reward_pre)
        return pre_gen_list, new_reward_dict
    
    def mutation(self, upgrade_input_list):
        upgrade_output_list = []
        for i in range(0, self.population_size):
            src_generate = copy.deepcopy(upgrade_input_list[i])
            single_generate = []
            for precision_index in src_generate:
                if np.random.rand() > self.random_best_update_filter_probility:
                    # rand_index = np.argmax(np.random.rand(len(self.SEARCH_SPACE[0])))
                    # while rand_index == precision_index:
                    #     rand_index = np.argmax(np.random.rand(len(self.SEARCH_SPACE[0])))
                    rand_index = self.SEARCH_SPACE[0][0] \
                        if precision_index == self.SEARCH_SPACE[0][1] else self.SEARCH_SPACE[0][1]
                    single_generate.append(rand_index)
                else:
                    single_generate.append(precision_index)
            upgrade_output_list.append(single_generate)
        return upgrade_output_list

    def crossover(self, upgrade_in_list, upgrade_out_list):
        cross_out_list = []
        for i in range(0, self.population_size):
            single_generate = []
            for j in range(0, len(self.SEARCH_SPACE)):
                single_generate.append(upgrade_out_list[i][j] \
                    if (random.random() <= self.cross_over_update_filter_probility) \
                        else upgrade_in_list[i][j])          
            cross_out_list.append(single_generate)
        return cross_out_list
    
    def random_generate(self):
        gen_set_list = []
        for space in self.SEARCH_SPACE:
            if self.random_bais == 8:
                prob = np.array([0.7, 0.3])
                _temp_rand = np.random.choice([0, 1], p=prob.ravel())
            elif self.random_bais == 16:
                prob = np.array([0.3, 0.7])
                _temp_rand = np.random.choice([0, 1], p=prob.ravel())
            else:
                _temp_rand = np.random.rand(len(space))
            gen_set_list.append(np.argmax(_temp_rand))
        return gen_set_list