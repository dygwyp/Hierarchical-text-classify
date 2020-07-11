# -*- coding: utf-8 -*-
# @Author: bruce
# @Date:   2019-05-05 19:00:28
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-07-23 21:59:33
# @Email: talkwithh@163.com
'''
文本分类标引实验。
方法：hierarchical && one vs all && stacking
通过组合多个二分类器，并训练单个类的10个分类器为1个分类器，最后测试。
训练包括2个部分：训练第一层分类器，训练第二层分类器，其中包括二分分类器和融合分类器
测试部分包括：测试二分类器；测试融合后的分类器；最终测试
数据来源：期刊文献。大小：约680MB，类别数量：不少于200，每个类别数量：500-2000。
'''
import random
import os,sys,time
from generate_data import generate_data,read_data
from level_1_train import Train_fasttext
from train_w2v import load_w2v_model
from pre_process import extract_data,pre_process,convert_tfidf,convert_w2v,save_pre_file
from level_3_train import train_binary_classifier,train_merge_classifier
from new_test import test

def main():
    # get the project params
    param_dic = get_param()
    base_path = param_dic['0']
    print('='*80)

    # generate data from original data
    print('I.抽取原始数据')
    con1,con2 = '',''
    params = param_dic['I']
    stopword_path,ori_1,ori_2,ch_file_path,target_path = params[0],params[1],params[2],params[3],params[4]
    print('I-0.读取原始数据')
    # con1 = read_data(ori_1)
    # con2 = read_data(ori_2)
    # if len(con1):
    #     re = con1.append(con2,ignore_index=True)
    # else:
    #     re = con2
    # generate_data(stopword_path,re,ch_file_path,target_path)
    print('='*80)
    print('='*80)

    # train level 1 classifier
    print('II.训练一级分类器')
    params = param_dic['II']
    data_path,train_save_path,test_save_path,model_save_path,result_save_path = params[0],params[1],params[2],params[3],params[4]

    fasttext_train = Train_fasttext(data_path,train_save_path,test_save_path,model_save_path,result_save_path)
    # fasttext_train.train_classifier()
    print('='*80)
    print('='*80)

    # train word2vec
    '''
    print('III-0.训练词向量')
    params = param_dic['III-0']
    '''
    # get dataset
    params = param_dic['III']
    print('III.训练三级分类器')
    print('III-1.获取类别数据')
    dataset_path,train_dataset_path = params[0],params[1]
    # extract_data(dataset_path,train_dataset_path)
    print('='*80)

    # pre-processing; generating train dataset and test dataset
    print('III-2.数据预处理')
    pre_process_path = base_path[-1]
    binary_data_path = pre_process_path+'binary_data_big.txt'
    train_merge_path = pre_process_path+'train_merge_data_big.txt'
    test_final_path = pre_process_path+'test_final_data_big.txt'
    # binary_data,train_merge_data,test_final_data = pre_process(train_dataset_path,pre_process_path)
    # 保存预处理文件
    print('保存文件..')
    # save_pre_file(binary_data_path,binary_data)
    # save_pre_file(train_merge_path,train_merge_data)
    # save_pre_file(test_final_path,test_final_data)
    print('='*80)

    # train binary classifier
    print('III-3.训练二分分类器')
    algorithm_list,w2v_model_path,level_3_model_save_path,level_3_result_save_path = params[2],params[3],params[4],params[5]
    # 选择算法（sgd,svm,bayes）
    algorithm = 'svm'
    # algorithm = get_algorithm(algorithm_list)
    # 文本表示
    # binary_data_dic = convert_tfidf(binary_data_path)
    w2v_model_path = w2v_model_path+'sgns.baidubaike.bigram-char'
    # w2v_model_path = w2v_model_path+'sgns.merge.bigram'
    # w2v_model_name = 'self_model/w2v_iter_5_model'
    w2v_save_path = pre_process_path +'binary_w2v_data_baidu.txt'
    # 加载w2v模型
    print('加载w2v模型..')
    w2v_model = load_w2v_model(w2v_model_path)
    # skip_word_save_path = level_3_result_save_path+'binary_result/'
    skip_word_save_path = ''
    # convert_w2v(binary_data_path,w2v_model,w2v_save_path,skip_word_save_path)
    
    # 训练分类器
    # train_binary_classifier(algorithm,w2v_save_path,level_3_model_save_path,level_3_result_save_path)
    print('='*80)
        
    # train a merge classifier from binary-classifiers
    print('III-4.训练融合分类器')
    # train_merge_classifier(algorithm,train_merge_path,w2v_model,level_3_model_save_path,level_3_result_save_path)
    print('='*80)

    # test all classifier
    print('III-5.测试分类系统')
    KB_fn = params[5]
    test(algorithm,test_final_path,level_3_result_save_path,level_3_model_save_path,w2v_model,skip_word_save_path,fasttext_train)

def get_algorithm(algorithm_list):
    str_p = 'please input model number'
    model = int(input(str_p+'(1-svm,2-bayes,3-sgd):'))
    if str(model).isdigit() and model != 0:
        algorithm = algorithm_list[model-1]
    return algorithm

def get_param():
    # 路经字典和列表
    param_dic = {}
    param_list = []

    '''基本路经参数'''
    # 原始数据路经
    ori_data_path = '../data/original_data/'
    # 信息路经
    info_path = '../data/info/'
    # 基本训练集和测试集路经
    train_path = '../data/train_dataset/'
    test_path = '../data/test_dataset/'
    # 基本模型路经
    model_path = '../data/model/'
    # 基本结果保存路径
    result_path = '../data/result/'
    # 预处理数据保存路径
    pre_process_path = '../data/pre_data/' # 原，未加temp
    param_list = [ori_data_path,info_path,train_path,test_path,model_path,result_path,pre_process_path]
    param_dic['0'] = param_list

    '''抽取数据路经参数'''
    # 停用词表路径
    stopword_path = info_path+'stopword/中文.txt'
    # 原始文件路径
    ori_path1 = ori_data_path+'ori_data/data1000_1.txt'
    ori_path2 = ori_data_path+'ori_data/data1000_2.txt'
    # 类目表路径
    le_file_path = info_path+'category/classify_list.xlsx'
    # 抽取数据保存路径
    target_path = ori_data_path+'target_path_le2/'
    param_list = [stopword_path,ori_path1,ori_path2,le_file_path,target_path]
    param_dic['I'] = param_list

    '''训练一层分类器参数'''
    # 训练数据路径
    level_1_data_path = target_path+'le2_data.txt'
    # 训练和测试数据保存路径
    level_1_train_save_path = train_path+'level_2/'
    level_1_test_save_path = test_path+'level_2/'
    # 模型和结果保存路经
    level_1_model_save_path = model_path   #原+'level_2/'
    level_1_result_save_path = result_path+'level_2/'
    param_list = [level_1_data_path,level_1_train_save_path,level_1_test_save_path,level_1_model_save_path,level_1_result_save_path]
    param_dic['II'] = param_list

    '''训练词向量参数'''
    train_data_path = target_path+'le1_data_w2v.txt'
    w2v_model_path = model_path+'W2V/'

    '''训练二层分类器参数'''
    dataset_path = target_path
    train_dataset_path = train_path + 'level_3_big/'
    # 模型和结果保存路经
    level_3_model_save_path = model_path+'level_3_big/'
    level_3_result_save_path = result_path+'level_3_big/'
    KB_fn = '../data/info/KB_json_of_data2'
    algorithm_list = ['svm','bayes','sgd']
    param_list = [dataset_path,train_dataset_path,algorithm_list,w2v_model_path,level_3_model_save_path,level_3_result_save_path,KB_fn]
    param_dic['III'] = param_list

    return param_dic

if __name__ == '__main__':
    main()
    