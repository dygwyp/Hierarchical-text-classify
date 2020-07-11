# -*- coding: utf-8 -*-
# @Author: bruce
# @Date:   2019-05-15 10:05:17
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-10-21 09:33:23
# @Email: talkwithh@163.com
'''
预处理数据
'''
import os,sys
import time
import random
import json
import numpy as np
from tool.json_encoder import MyEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from train_w2v import get_train_vec,load_w2v_model
from read_write_tool import read_file,save_file,read_file_lines,save_file_lines

NUMBER = 10  #每个类训练的二分类器数目
RANGE_SIZE_OF_DATA = [500,5000]  # 每个类抽取数据条数范围
SCALE_OF_DATA = [0.5,0.4,0.1]   # 二分类训练集、融合训练集和测试集比例

# 从数据集中抽取出三级类别的文档并保存
def extract_data(original_data_path,save_data_path):
    start_time = time.time()
    # 获取统计数据
    # 统计所有三级类别数量
    cate_num = 0
    # 获得一级类别列表
    cate_list = [file_dir for file_dir in os.listdir(original_data_path) if os.path.isdir(original_data_path+file_dir)]
    print('..level1 类别：'+str(cate_list))
    for cate in cate_list:
        cur_path = original_data_path + cate + '/'
        file_list = os.listdir(cur_path)
        # 获得统计文件和文本文件全路径
        count_file = cur_path + [file for file in file_list if 'count' in file][0]
        text_file = cur_path + [file for file in file_list if not 'count' in file][0]

        count_con = read_file_lines(count_file)
        target_cate = [cate.strip().split('-->')[0] for cate in count_con if int(cate.split('-->')[1]) >= RANGE_SIZE_OF_DATA[0]]
        cate_num += len(target_cate)
        data_list = read_file_lines(text_file)

        save_path = save_data_path + cate +'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # content_dic = dict(zip(target_cate,[[]]*len(target_cate)))    # 此种方法构建字典，添加数据出现BUG
        content_dic = {}
        for data in data_list:
            data = data.replace('\r','')
            long_label = data.split(',')[0]
            index = long_label.index('__',2)
            level3_cate = long_label[(index+2):]
            if level3_cate in target_cate:
                if not level3_cate in content_dic.keys():
                    content_dic[level3_cate] = []
                    content_dic[level3_cate].append(data)
                else:
                    content_dic[level3_cate].append(data)
        # print('write file %s...'%(cur_path))
        for level3_cate in target_cate:
            save_file_lines(save_path+level3_cate+'_data.txt',content_dic[level3_cate][:RANGE_SIZE_OF_DATA[1]],'w')

    print('..level3 类别总数：{}'.format(cate_num))
    end_time = time.time()
    print('生成基本训练集用时：{}'.format(end_time-start_time))
# pre-processing dataset.
# step1: extract data randomly. eg: A(50),-A(50). total:10 dataset of A
# step2: convert into a vector of tf-idf.
def pre_process(demo_data_path,pre_process_path):
    # step1 extract data
    print('..生成训练测试数据')
    start_time = time.time()
    if not os.path.exists(pre_process_path):
        os.makedirs(pre_process_path)
    # the scale of data: 5:4:1
    # train dataset(train_binary_data) for binary classifier: 500 records.
    # binary_data_dic for a sub-dataset for one categpry. eg:A(50),-A(50)
    train_binary_data,binary_data_dic = {},{}
    # train dataset for merge classifier: 400 records.
    train_merge_data = {}
    # test dataset finally: 100 records
    test_final_data = {}
    # get category list
    cate_3_list = []
    cate_list = [cate for cate in os.listdir(demo_data_path) if not cate.startswith('.')]    
    # print('level-1 类别列表：'+str(cate_list))
    cate_length = len(cate_list)
    for cate in cate_list:
        if 'txt' in cate:
            continue
        file_path = demo_data_path + cate + '/'
        for file in os.listdir(file_path):
            flag = 0
            if file.startswith('.'):
                continue
            if '_' in file:
                file_name_list = file[:-4].split('_')   # file[:-4] 表示去掉‘.txt’
                cate_3 = file_name_list[0]
                cate_3_num = file_name_list[-1]                                 # cate_3 = file[:file.index('_')]
                if int(cate_3_num) >= RANGE_SIZE_OF_DATA[0] and int(cate_3_num) <= RANGE_SIZE_OF_DATA[1]:        # 抽取大于500数据的三级类
                    cate_3_list.append(cate_3)
                    file_full_path = file_path + file
                    content = read_file_lines(file_full_path)
                    flag = 1
                elif int(cate_3_num) > RANGE_SIZE_OF_DATA[1]:       # 大于5000数据的三级类只抽取5000
                    cate_3_list.append(cate_3)
                    file_full_path = file_path + file
                    content = read_file_lines(file_full_path)[:RANGE_SIZE_OF_DATA[1]]
                    flag = 1
            if not flag:
                continue
            con_length = len(content)
            train_size = int(con_length*SCALE_OF_DATA[0])
            merge_size = int(con_length*SCALE_OF_DATA[1])
            test_size = int(con_length*SCALE_OF_DATA[2])


            train_binary = content[:train_size]
            train_merge = content[train_size:(train_size+merge_size)]
            if test_size >= 100:
                test_data = content[(train_size+merge_size):(train_size+merge_size+100)]
            else:
                test_data = content[(train_size+merge_size):]

            # 随机抽取属于本类别的数据，100条
            train_binary_self = []
            # 随机抽取用于其它类别训练的数据，100条
            train_binary_other = []
            for i in range(NUMBER):
                train_binary_self.append(random.sample(train_binary,2*train_size//NUMBER))
                train_binary_other.append(random.sample(train_binary,2*train_size//(NUMBER)))
            train_binary_data[cate_3] = [train_binary_self,train_binary_other]
            
            # 抽取用于训练融合分类器的数据，800条
            train_merge_data[cate_3] = train_merge
            # 抽取测试数据,200条
            test_final_data[cate_3] = test_data
            # 初始化sub-dataset
            binary_data_dic[cate_3] = []
    # generate train sub-dataset
    class_number = len(cate_3_list)
    print('三级类数目：{}\n'.format(class_number))
    
    for i in range(class_number):
        cur_cate = cate_3_list[i]
        for m in range(NUMBER):
            self_data = train_binary_data[cur_cate][0][m]
            other_data = [] # 其它数据初始化
            for j in range(NUMBER):
                other_cate = cate_3_list[(j+1)%class_number]   # 原：(j*5+class_number//NUMBER)%class_number
                if other_cate == cur_cate:    # 加入层级分类时，将其他类的抽取方式改为抽取本大类下的数据
                    continue
                elif len(other_data) and (len(other_data) > len(self_data)):
                    break
                else:
                    other_data += train_binary_data[other_cate][1][m]
            cur_con = self_data+other_data
            self_con,self_label = get_dataset(cur_con,cur_cate)

            binary_data_dic[cur_cate].append([self_con,self_label])
   
    end_time = time.time()
    print('数据预处理用时：{}s'.format(end_time-start_time))
    return binary_data_dic,train_merge_data,test_final_data

def save_pre_file(file_path,file_con):
    save_file(file_path,file_con,'w')

def load_data(binary_data_path):
    start_time = time.time()
    binary_data_dic = eval(read_file(binary_data_path))
    end_time = time.time()
    print('读取数据用时：{}s'.format(end_time-start_time))

    return binary_data_dic

def convert_tfidf(binary_data_path):
    print('..tf-idf文本表示')
    start_time = time.time()
    binary_data_dic = load_data(binary_data_path)
     # 转换为tf-idf向量
    for cate,data in binary_data_dic.items():
        for i in range(NUMBER):
            self_con,self_label = binary_data_dic[cate][i][0],binary_data_dic[cate][i][1]
            vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5)
            self_tdm = vectorizer.fit_transform(self_con)
            vocabulary = vectorizer.vocabulary_
            binary_data_dic[cate][i] = [self_tdm,self_label,vocabulary]
            # binary_data_dic[cur_cate].append([self_tdm,self_label,vocabulary])
    end_time = time.time()
    print('文本表示用时：{}s'.format(end_time-start_time))
    return binary_data_dic

def convert_w2v(binary_data_path,w2v_model,binary_w2v_data_path,skip_save_path=''):
    print('..word2vec文本表示')
    start_time = time.time()
    # 加载数据
    binary_data_dic = load_data(binary_data_path)
    # 加载w2v模型
    # w2v_model = load_w2v_model(w2v_model_path)
    # load_model_time = time.time()
    # print('加载w2v模型用时：{}s'.format(load_model_time-start_time))
    for cate,data in binary_data_dic.items():
        print('cur_cate: '+cate)
        for i in range(NUMBER):
            self_con,self_label = binary_data_dic[cate][i][0],binary_data_dic[cate][i][1]
            train_w2v = get_train_vec(self_con,w2v_model,skip_save_path)
            binary_data_dic[cate][i] = [train_w2v,self_label]

    doc_w2v_json = json.dumps(binary_data_dic,cls=MyEncoder)      # 速度比上一行快
    save_file(binary_w2v_data_path,doc_w2v_json,'w')
    end_time = time.time()
    print('文本表示用时：{}s'.format(end_time-start_time))
    return binary_data_dic

def get_dataset(data_set,cur_cate):
    label = []
    content = []
    for item in data_set:
        item_list = item.split(',')
        content.append(item_list[1].strip())
        # cur_label = item_list[0].split('_')[-1][0]
        index = item_list[0].index('__',2)
        cur_label = item_list[0][(index+2):]
        if cur_label == cur_cate:
            label.append(cur_label)
        else:
            label.append('-'+cur_cate)
    return content,label
