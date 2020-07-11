# -*- coding: utf-8 -*-
# @Author: bruce
# @Date:   2019-05-02 22:31:09
# @Last Modified by:   bruce
# @Last Modified time: 2019-05-31 13:50:52
# @Email: talkwithh@163.com
'''
训练fasttext中图一级分类器，
'''
import os,sys,time
# sys.path.append("..")
import numpy as np
# import fasttext    #os/linux系统
import fastText.FastText as ff  #win系统
from sklearn.model_selection import train_test_split
from read_write_tool import read_file_lines,save_file_lines,save_file

class Train_fasttext():
    # Initialization
    def __init__(self,data_path,train_save_path,test_save_path,model_save_path,result_save_path):
        self.data_path = data_path
        self.train_save_path = train_save_path
        self.test_save_path = test_save_path
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.train_save_path):
            os.makedirs(self.train_save_path)
        if not os.path.exists(self.test_save_path):
            os.makedirs(self.test_save_path)
        if not os.path.exists(self.result_save_path):
            os.makedirs(self.result_save_path)

    # load dataset
    def load_train_dataset(self):
        data_list = read_file_lines(self.data_path)
        con,label = [],[]
        for line in data_list:
            line_list = line.split(',')
            label.append(line_list[0])
            con.append(line_list[1])

        x_train,x_test,y_train,y_test = train_test_split(con,label,test_size=0.4)#default test_size=0.25

        train_con,test_con = [],[]
        # train_con = self.merge_con_label(x_train,y_train)
        test_con = self.merge_con_label(x_test,y_test)
        # all_con = self.merge_con_label(con,label)

        # save_file_lines(self.train_save_path+'train.txt',train_con)
        save_file_lines(self.test_save_path+'test_big.txt',test_con)
        # save_file_lines(self.train_save_path+'all_data.txt',all_con)        

    # merge the content and label
    def merge_con_label(self,con,label):
        merge_list = []
        for x,y in zip(con,label):
            merge_list.append(y+', '+x)

        return merge_list
    # train single classifier. number:one
    # classify the all categories
    def train_classifier(self):
        # self.load_train_dataset()
        #实验后的最佳参数之一
        
        start_time = time.time()
        classifier=ff.train_supervised(self.data_path,lr=0.1,loss='hs',wordNgrams=2,epoch=300)# epoch=20,0.91;epoch=50,0.93; 
        model = classifier.save_model(self.model_save_path+'level_2_fasttext_classifier_big_big.model') # 保存模型  all:0.91;all_2:0.93
        classifier.get_labels() # 输出标签
        # 测试模型
        # print('加载fasttext模型--{}'.format('level_1_fasttext_classifier_big_test.model'))
        # classifier = ff.load_model(self.model_save_path+'level_1_fasttext_classifier_big_test.model')
        test_result = classifier.test(self.test_save_path+'test_big.txt')
        result_str = 'test precision:{}\n'.format(test_result)
        print(result_str)

        end_time = time.time()
        load_time = round(end_time-start_time,3)
        train_time_str = 'train and test model time %fs'%load_time
        print(train_time_str)

        save_file(self.result_save_path+'fasttext_result_big.txt',result_str+train_time_str+'\n','a')
        
    # test model
    def test_model(self,input_text,classifier):
        # 判断模型是否训练完成
        result = classifier.predict(input_text)
        # print('预测一级类别：%s'%(result[0]))
        return result[0]

    def load_fasttext(self,model_name):
        start_time = time.time()
        classifier = ff.load_model(self.model_save_path+model_name)
        end_time = time.time()
        pre_time = round(end_time-start_time,3)
        print('加载fasttext模型时间: %f'%pre_time)
        return classifier

if __name__ == '__main__':
    data_path = '../data/original_data/target_path_le2/le2_data.txt'
    train_save_path = '../data/train_dataset/level_2/'
    test_save_path = '../data/test_dataset/level_2/'
    model_save_path = '../data/model/level_2/'
    result_save_path = '../data/result/level_2/'
    # train = Train_fasttext(data_path,train_save_path,test_save_path,model_save_path,result_save_path)
    # train.train_classifier()
    input_text = ['运用 栅格 矩阵 建立 兵棋 地图 地形 属性 兵棋 地图','我国 进行 作用 油溶性 工业生产 缓蚀剂 综述 环保型 物质 重要 同时 进一步 重点 调控 应用 最佳 金属 阐述 组合 基体 核心 防腐 研究 化学工程 油脂 有利于 机制 现状 金属材料 结合']
    # 第二个文本是T类
    # classifier = train.load_fasttext()
    # train.test_model(input_text,classifier)

    # data_path = 'D:/Data/data1&2_trainingdata_of_fasttext/level_1.txt'
    data_path = 'D:/Data/data1&2_trainingdata_of_fasttext/level_2.txt'
    
    train = Train_fasttext(data_path,train_save_path,test_save_path,model_save_path,result_save_path)
    train.train_classifier()
