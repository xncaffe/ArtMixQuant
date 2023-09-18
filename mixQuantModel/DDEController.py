# -*- coding: utf-8 -*-
import numpy as np
import random
import copy
import os
import operator
from shutil import copyfile
from mixQuantEngine.artRuntime import *
class DiscreteDifferentialEvolutionController(object):
    def __init__(self,MIND=100,M=0.5,XOVR=0.7,SEARCH_SPACE=None,bigger_is_better=False,random_bais=0):
        self.MIND=MIND
        self.M=M
        self.XOVR=XOVR
        self.SEARCH_SPACE=SEARCH_SPACE
        self.warmup_flg=False
        self.generation_counter=0
        self.current_generations=[]
        self.sample_generations_regist=[]
        self.new_generations=[]
        self.new_generations_regist=[]
        self.reward_dict={}
        self.random_bais=random_bais
        self.init_generation()
        self.bigger_is_better=bigger_is_better
        self.generation_best_record=[]
        self.loss_record=[]
        self.remember_best_num=10
        self.best_record=[]
        self.init_performance=[]

    def init_generation(self):
        for i in range(self.MIND):
            g=self.random_generate()
            self.current_generations.append(g)
            self.sample_generations_regist.append(g)
            g=self.random_generate()
            self.new_generations.append(g)
            self.sample_generations_regist.append(g)

    def get_newparam(self):
        if(len(self.sample_generations_regist)>0):
            ret_param=self.sample_generations_regist[0]
            self.sample_generations_regist=self.sample_generations_regist[1:]
            return ret_param
        else:
            self.current_generations,self.reward_dict=self.selection(self.new_generations,self.current_generations)
            self.record_best(self.reward_dict)
            self.new_generations=self.gen_new_generation(self.current_generations)
            self.sample_generations_regist=self.new_generations.copy()
            ret_param=self.sample_generations_regist[0]
            self.sample_generations_regist=self.sample_generations_regist[1:]
            return ret_param

    def record_best(self,m):
        if self.bigger_is_better:
            best_value=max(m.values())
        else:
            best_value=min(m.values())
        for key,value in m.items():
            if(value == best_value):
                self.generation_best_record.append([key,value])
                self.loss_record.append(value)

    def dictkey2list(self,reward_dict):
        l=[]
        for key in reward_dict:
            l.append(list(key))

    def update_reward(self,key,value,cycle,rmse,relDistance,test_name,JsonDirSturct):
        self.reward_dict.setdefault(tuple(key), value)
        if rmse<=self.init_performance[0]["rmse"]:
            record={"test":test_name,"reward":value,"cycle":cycle,"rmse":rmse,"relDistance":relDistance,"sample":key}
            bestMixJsonDir=os.path.join(JsonDirSturct["mixJsonDir"],"bestMixJson")
            if not os.path.exists(bestMixJsonDir):
                create_dir(bestMixJsonDir)
            mainJsonFile=os.path.join(JsonDirSturct["mixJsonDir"],get_ini_name(JsonDirSturct["ini_mix_path"])+"_"+str(test_name)+".json")
            bestJsonFile=os.path.join(bestMixJsonDir,get_ini_name(JsonDirSturct["ini_mix_path"])+"_"+str(test_name)+".json")
            if len(self.best_record)<self.remember_best_num:
                self.best_record.append(record)
                copyfile(mainJsonFile,bestJsonFile)
                self.best_record=sorted(self.best_record, key=operator.itemgetter('relDistance'))
            else:
                best_len=len(self.best_record)
                for _id in range(best_len):
                    _record=self.best_record[_id]
                    if relDistance<=_record["relDistance"]:
                        resort=False
                        if(relDistance==_record["relDistance"] and cycle==_record["cycle"]):
                            resort=True
                            delJsonFile=os.path.join(bestMixJsonDir,get_ini_name(JsonDirSturct["ini_mix_path"])+"_"+str(self.best_record[_id]["test"])+".json")
                        else:
                            delJsonFile=os.path.join(bestMixJsonDir,get_ini_name(JsonDirSturct["ini_mix_path"])+"_"+str(self.best_record[-1]["test"])+".json")
                            for _nx in range(best_len-_id-1):
                                self.best_record[best_len-_nx-1]=self.best_record[best_len-_nx-2]
                        if os.path.exists(delJsonFile):
                            os.remove(delJsonFile)
                        self.best_record[_id]=record
                        copyfile(mainJsonFile,bestJsonFile)
                        if resort:
                            self.best_record=sorted(self.best_record, key=operator.itemgetter('relDistance'))
                        break

    def gen_new_generation(self,np_list):
        v_list=self.mutation(np_list)
        return self.crossover(np_list,v_list)

    def selection(self,u_list,np_list):
        new_reward_dict={}
        for i in range(0,self.MIND):
            key_u=tuple(u_list[i])
            reward_u=self.reward_dict[key_u]
            key_np=tuple(np_list[i])
            reward_np=self.reward_dict[key_np]
            if self.bigger_is_better:
                if reward_np <= reward_u:
                    np_list[i] = u_list[i]
                    new_reward_dict.setdefault(tuple(key_u), reward_u)
                else:
                    new_reward_dict.setdefault(tuple(key_np), reward_np)
            else:
                if reward_np >= reward_u:
                    np_list[i] = u_list[i]
                    new_reward_dict.setdefault(tuple(key_u), reward_u)
                else:
                    new_reward_dict.setdefault(tuple(key_np),reward_np)
        return np_list,new_reward_dict

    def mutation(self,np_list):
        v_list = []
        for i in range(0,self.MIND):
            ori_sample=copy.deepcopy(np_list[i])
            vv_list = []
            for ele in ori_sample:
                if np.random.rand()>self.M:
                    _temp_rand=np.argmax(np.random.rand(len(self.SEARCH_SPACE[0])))
                    while _temp_rand==ele:
                        _temp_rand=np.argmax(np.random.rand(len(self.SEARCH_SPACE[0])))
                    vv_list.append(_temp_rand)
                else:
                    vv_list.append(ele)
            v_list.append(vv_list)
        return v_list

    def crossover(self,np_list,v_list):
        u_list = []
        for i in range(0,self.MIND):
            vv_list = []
            for j in range(0,len(self.SEARCH_SPACE)):
                # if (random.random() <= self.XOVR) | (j == random.randint(0,len(self.SEARCH_SPACE) - 1)):
                if (random.random() <= self.XOVR) :
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            u_list.append(vv_list)
        return u_list
    
    def random_generate(self):
        g=[]
        for space in self.SEARCH_SPACE:
            if self.random_bais==8:
                prob=np.array([0.7, 0.3])
                _temp_rand=np.random.choice([0,1], p=prob.ravel())
                g.append(_temp_rand)
            elif self.random_bais==16:
                prob=np.array([0.3, 0.7])
                _temp_rand=np.random.choice([0,1],p=prob.ravel())
                g.append(_temp_rand)
            else:
                _temp_rand=np.random.rand(len(space))
                g.append(np.argmax(_temp_rand))
        return g

