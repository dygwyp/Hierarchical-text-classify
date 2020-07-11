# -*- coding: utf-8 -*-
# @Author: bruce·li
# @Date:   2019-10-19 09:44:46
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-10-21 09:52:36
# @Email:   talkwithh@163.com
'''
训练分类器
'''
import os,time
import random
import json
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from algorithm_model import SVM_model,bayes_model,sgd_model
from pre_process import get_dataset
from train_w2v import get_train_vec,load_w2v_model
from tool.json_encoder import MyEncoder
from read_write_tool import read_file,save_file,read_file_lines,save_file_lines,load_model,save_model


def main(original_data_path,datalevel_info_file,w2v_model_path,dataset_w2v_data_path):
    # 获取数据集
    get_dataset(original_data_path,datalevel_info_file,w2v_model_path,dataset_w2v_data_path)

    # 训练分类器

    # 测试分类系统
    

def train_single_classifier():
    pass

def test():
    pass

def get_dataset(original_data_path,datalevel_info_file,w2v_model_path,dataset_w2v_data_path):
    dataset_dic = {}
    dataset_label_dic = {}

    # 加载词向量
    # w2v_model = None
    w2v_model = load_w2v_model(w2v_model_path)
    # 读取数据集二级类目信息
    infos = read_file_lines(datalevel_info_file)
    level2_list = eval(infos[0])
    level2_dic = eval(infos[1])
    # 初始化数据集字典
    for le2 in level2_list:
        if le2 not in dataset_dic:
            dataset_dic[le2] = []
            dataset_label_dic[le2] = []

    # 读取数据集
    data_list,data_label_list = [],[]
    contents = read_file_lines(original_data_path+'le3_data.txt')
    for line in contents:
        line_list = line.split(',')
        label = line_list[0].replace('__label__','')
        content = line_list[1]

        if label[:3] in level2_list:
            dataset_dic[label[:3]].append(content)
            dataset_label_dic[label[:3]].append(label)
        elif label[:2] in level2_list:
            dataset_dic[label[:2]].append(content)
            dataset_label_dic[label[:2]].append(label)
    # 保存中间文件
    dataset_w2v_dic = {}
    key_list = dataset_dic.keys()
    for key in key_list:
        # if not len(dataset_dic[key]):
        #     print(key)
        if key not in dataset_w2v_dic:
            dataset_w2v_dic[key] = []
        dataset_w2v_dic[key] = get_train_vec(dataset_dic[key],w2v_model)

    doc_w2v_json = json.dumps(dataset_w2v_dic,cls=MyEncoder)
    save_file(dataset_w2v_data_path,doc_w2v_json,'w')

if __name__ == '__main__':
    original_data_path = '../data/original_data/target_path/'
    datalevel_info_file = '../data/original_data/datalevel_info.txt'
    w2v_model_path = '../data/model/sgns.baidubaike.bigram-char'
    dataset_w2v_data_path = '../data/dataset/level3/single/dataset_w2v_data.txt'
    main(original_data_path,datalevel_info_file,w2v_model_path,dataset_w2v_data_path)