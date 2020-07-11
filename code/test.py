# -*- coding: utf-8 -*-
# @Author: bruce
# @Date:   2019-05-18 09:29:18
# @Last Modified by:   bruce
# @Last Modified time: 2019-05-18 16:51:54
# @Email: talkwithh@163.com
'''
测试分类器
'''
import os,time
import numpy as np
from pre_process import get_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from read_write_tool import read_file,save_file,read_file_lines,save_file_lines,load_model,save_model

SINGLE_CLASS_SIZE = 1000 # 抽取数据条数阈值
NUMBER = 10  #每个类训练的二分类器数目
test_size = 100     # 单个类测试数据量
SCALE_OF_DATA = [0.5,0.4,0.1]   # 二分类训练集、融合训练集和测试集比例

def test(algorithm,test_final_data,result_save_path,model_save_path):
    cate_list = list(test_final_data.keys())
    class_number = len(cate_list)
    test_start = time.time()
    # load model
    model_dic = {}
    model_merge_dic = {}
    for j in range(class_number):
        cur_cate = cate_list[j]
        model_dic[cur_cate] = []
        model_path = model_save_path+cur_cate+'/'
        models = os.listdir(model_path)
        for model in models:
            if algorithm in model:      # svm
                model_full_path = model_path + model
                model_dic[cur_cate].append(model_full_path)
    for model in os.listdir(model_save_path):
        if algorithm in model:
            cate = model.split('_')[0]
            model_merge_dic[cate] = load_model(model_save_path+model)

    model_name_map = eval(read_file(result_save_path+algorithm+'_model_name_map.txt'))

    all_right = 0
    all_len = 0
    result_path = result_save_path+'final_test'+'/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    test_result_path = result_path+algorithm+'_test.txt'

    test_datas = {}
    error_cate = {}
    classify_cate = {}
    # 得到测试文本和标签
    for i in range(class_number):
        cur_cate = cate_list[i]
        contents,labels = get_dataset(test_final_data[cur_cate],cur_cate)
        test_datas[cur_cate] = [contents,labels]
    # 测试过程
    for cate,cons in test_datas.items():
        test_one_time = time.time()
        cur_cate = cate
        print('cur cate %s'%cur_cate)
        texts,labels = cons[0],cons[1]
        right = 0
        doc_dic = {}
        error_cate[cate] = []
        classify_cate[cate] = []

        test_size = len(labels)
        all_len += test_size
        
        doc_dic = {i:[] for i in range(test_size)}
        for bin_cate,models in list(model_dic.items()):
            # if not (bin_cate == 'A81' or bin_cate == 'B08' or bin_cate == 'D80'):
            #     continue
            text_pro = []
            pro_matrix = np.array([],[])
            # print('load binary model %s'%bin_cate)
            for model in models:
                clf = load_model(model)
                voc = model_name_map[model]
                vectorizer = TfidfVectorizer(vocabulary=voc)
                tdm = vectorizer.fit_transform(texts)
                pred = clf.predict_proba(tdm)
                for i in range(len(pred)):
                    text_pro.append(pred[i][1])
            pro_matrix = np.array(text_pro).reshape((NUMBER,test_size)).T
          
            temp_pro = []
            for c,md in model_merge_dic.items():
                if bin_cate == c:
                    pre = md.predict_proba(pro_matrix)
                    for j in range(len(pre)):
                        doc_dic[j].append([pre[j][1],c])
                    
        # print(doc_dic)
        for doc,pro_list in doc_dic.items():
            pro_sort = sorted(pro_list,key=lambda d:d[0], reverse = True)
            pre_cate = [pro_sort[0][1],pro_sort[1][1],pro_sort[2][1]]       #选择top3预测的类别,,准去率0.40+
            if cur_cate in pre_cate:      #  旧方法 ：pro_sort[0][1] == cur_cate     
                right += 1
                all_right += 1
            else:
                error_cate[cate].append(pre_cate)
            classify_cate[cate].append(pre_cate)
            doc_dic[doc] = pro_sort[0][1]
            
        # print(doc_dic[2])
        acc = right/test_size
        print('acc %f'%acc)
        test_one_end_time = time.time()
        one_run_time = round(test_one_end_time-test_one_time,4)
        print('test one cate time:%f\n'%one_run_time)
        
        save_file_lines(test_result_path,cur_cate+' dataset accuracy :%f'%acc+'\n','a')
        save_file_lines(result_save_path+'final_test/'+algorithm+'_all.txt','\n'+cate+'\n'+str(classify_cate[cate]),'a')
        save_file_lines(result_save_path+'final_test/'+algorithm+'_error.txt','\n'+cate+'\n'+str(error_cate[cate]),'a')
        break
    # print('macro acc %f'%(all_right/all_len))
    # test_end = time.time()
    # test_run_time = round(test_end-test_start,4)
    # print('merge test time: %f'%(test_run_time))
    # save_file_lines(test_result_path,'using '+algorithm+' macro acc %f'%(all_right/all_len)+'\n','a')
    # save_file_lines(test_result_path,'merge test time: %f'%(test_run_time),'a')