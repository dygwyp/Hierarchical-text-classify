# -*- coding: utf-8 -*-
# @Author: bruce
# @Date:   2019-05-18 09:29:18
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-07-23 22:08:46
# @Email: talkwithh@163.com
'''
测试分类器
'''
import os,time
import numpy as np
import json
from pre_process import get_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from level_1_train import Train_fasttext
from level_3_train import get_model,load_binary_model
from train_w2v import get_train_vec,get_sent_vec,load_w2v_model
from get_level_3_from_KB_to_yx import get_level_3_from_KB,get_KB_dic
from read_write_tool import read_file,save_file,read_file_lines,save_file_lines,load_model,save_model

NUMBER = 10  #每个类训练的二分类器数目
SCALE_OF_DATA = [0.5,0.4,0.1]   # 二分类训练集、融合训练集和测试集比例

def load_merge_model(algorithm,model_save_path):
    model_merge_dic = {}
    for model in os.listdir(model_save_path):       
        if algorithm in model:
            le3_cate = model.split('_')[0]
            model_merge_dic[le3_cate] = model_save_path+model

    return model_merge_dic

def get_test_dataset(class_number,cate_list,test_final_data):
    test_data_dic = {}
    # 得到测试文本和标签
    for i in range(class_number):
        cur_cate = cate_list[i]
        contents,labels = get_dataset(test_final_data[cur_cate],cur_cate)
        test_data_dic[cur_cate] = [contents,labels]

    return test_data_dic

def test(algorithm,test_final_path,result_save_path,model_save_path,w2v_model,skip_word_save_path,fasttext_train):
    # 读取测试集文件
    test_final_data = eval(read_file(test_final_path))
    cate_list = list(test_final_data.keys())
    class_number = len(cate_list)
    test_start = time.time()
    
    # 加载二分类器映射文件
    # model_name_map = eval(read_file(result_save_path+algorithm+'_model_name_map_json.txt'))
    # model_name_map = json.loads(read_file(result_save_path+algorithm+'_model_name_map_json.txt'))   # 速度比上一行快

    all_len,all_right = 0,0
    result_path = result_save_path+'final_test'+'/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    record_path = result_path+'records_way_le1_le2_w2v/'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    test_result_path = result_path+algorithm+'_test_way_le1_le2_w2v.txt'
    
    # 得到测试文本和标签
    test_data_dic = get_test_dataset(class_number,cate_list,test_final_data)

    # 测试过程
    # 加载模型
    load_model_time = time.time()
    # load fasttext model(level_1)
    le1_model = 'level_1/level_1_fasttext_classifier_big_big.model'
    le1_fasttext_model = fasttext_train.load_fasttext(le1_model)
    le2_model = 'level_2/level_2_fasttext_classifier_big_big.model'
    le2_fasttext_model = fasttext_train.load_fasttext(le2_model)


    # get sklearn classifier model
    # clf = get_model(algorithm)
    # load binary models
    print('..加载二分类器模型')
    model_dic = load_binary_model(algorithm,class_number,cate_list,model_save_path)
    load_end_time = time.time()
    print('加载模型用时：{}'.format(load_end_time-load_model_time))
    # load merge model
    model_merge_dic = load_merge_model(algorithm,model_save_path+'merge_model/')
    # 加载w2v模型
    # w2v_model = load_w2v_model(w2v_model_path)

    # read KB
    # kb_dic = get_KB_dic(skip_word_save_path)

    for cate,cons in test_data_dic.items():
        test_one_time = time.time()
        cur_cate = cate
        print('cur cate %s'%cur_cate)

        # 定义TP，FP，TN，FN
        # tp_num = 0
        right = 0
        texts,labels = cons[0],cons[1]
        
        test_size = len(labels)
        all_len += test_size
        
        # 一级类目预测/二级
        level_1_pre_result = fasttext_train.test_model(texts,le1_fasttext_model)
        level_2_pre_result = fasttext_train.test_model(texts,le2_fasttext_model)
        level_1_pre_labels_list,level_2_pre_labels_list = [],[]
        for le1 in level_1_pre_result:
            label_list_le1 = le1[0][:-1].split('__')
            level_1_pre_labels_list.append(label_list_le1[2])

        for le2 in level_2_pre_result:
            label_list_le2 = le2[0][:-1].split('__')
            level_2_pre_labels_list.append(label_list_le2[2])

        # print(level_2_pre_labels_list)
        # 知识库预测
        # text_list = [text.split() for text in texts]
        # kb_labels_list = get_level_3_from_KB(kb_dic,text_list)
        
        # 记录
        text_pre_results = {}
        for i in range(test_size):
            text,label = texts[i],labels[i]
            text_pre_results[label+'\t'+text] = []

            # text_kb_label = []
            # text_kb_label = kb_labels_list[i]
            text_le2_label = []
            text_le1_label = level_1_pre_labels_list[i]
            text_le2_label = level_2_pre_labels_list[i]
            le_flag = 0
            if text_le2_label[0] == text_le1_label:
                le_flag = True
            # if label not in text_kb_label:  # 知识库预测结果未出现文档原始标签，直接跳过       way_5未跳过
            #     continue
            pre_result_dic = {}
            for bin_cate,models in model_dic.items():
                skip_flag = 0
                # if bin_cate not in text_kb_label:
                #     continue
                if not le_flag and bin_cate[0] in text_le1_label:
                    skip_flag = 1
                if le_flag and bin_cate[:2] in text_le2_label:
                    skip_flag = 2
                # if bin_cate[0] in text_le1_label:
                #     skip_flag = 3
                if skip_flag: 
                    text_pro = []
                    pre_result_dic[bin_cate] = []
                    for model in models:
                        # clf = load_model(model)
                        clf = model[1]
                        train_w2v = get_train_vec([text],w2v_model,skip_word_save_path)
                        # voc = model_name_map[model[0]]
                        # vectorizer = TfidfVectorizer(vocabulary=voc)
                        # tdm = vectorizer.fit_transform([text])
                        pred = clf.predict_proba(train_w2v)
                        text_pro.append(pred[0][1])
                        
                    for c,md in model_merge_dic.items():
                        if bin_cate in md:
                            # print('load merge model %s'%mer_md)
                            merge_model = load_model(md)
                            pre = merge_model.predict_proba([text_pro])
                            pre_result_dic[bin_cate].append(pre[0][1])
                    
                
            sort_pre_tuple = sorted(pre_result_dic.items(), key=lambda d:d[1],reverse=True)
            sort_cate_len = len(sort_pre_tuple)
            pre_cate_list = []
            for pre_cate in sort_pre_tuple:
                pre_cate_list.append(pre_cate[0])
            pre_len = len(pre_cate_list)
            if pre_len >=3:
                if label in pre_cate_list[:3]:
                    right += 1
                    all_right += 1
            elif label in pre_cate_list:
                right += 1
                all_right += 1

            text_pre_results[label+'\t'+text].append([text_le1_label,text_le2_label,pre_cate_list[:10]])
            
        acc = right/test_size
        print('acc %f'%acc) # 这种方法的准确率为0.21,6h--way_1  不加入fasttext; 加入fasttext效果提高到0.23/way_3,way_4
        save_file_lines(record_path+cur_cate+'_svm_result_record_way_le1_le2_w2v.txt',text_pre_results,'w')
        '''
        right = 0
        # doc_dic = {}
        doc_dic = {i:[] for i in range(test_size)}
        for bin_cate,models in model_dic.items():
            if bin_cate[0] not in level_2_pre_labels_list:
                continue
            text_pro = []
            pro_matrix = np.array([],[])
            # print('load binary model %s'%bin_cate)
            for model in models:
                # clf = load_model(model)
                clf = model[1]
                voc = model_name_map[model[0]]
                vectorizer = TfidfVectorizer(vocabulary=voc)
                tdm = vectorizer.fit_transform(texts)
                pred = clf.predict_proba(tdm)
                for i in range(len(pred)):
                    text_pro.append(pred[i][1])
            pro_matrix = np.array(text_pro).reshape((NUMBER,test_size)).T
            temp_pro = []
            for c,md in model_merge_dic.items():
                if bin_cate in md:
                    # print('load merge model %s'%mer_md)
                    merge_model = load_model(md)
                    pre = merge_model.predict_proba(pro_matrix)
                    for j in range(len(pre)):
                        doc_dic[j].append([pre[j][1],c])
        # print(doc_dic)
        # 不加入KB和层级分类
        
        for doc,pro_list in doc_dic.items():
            pro_sort = sorted(pro_list,key=lambda d:d[0], reverse = True)
            pre_cate = [pro_sort[0][1],pro_sort[1][1],pro_sort[2][1]]       #选择top3预测的类别,,准去率0.40+
            if cur_cate in pre_cate:      #  旧方法 ：pro_sort[0][1] == cur_cate     
                right += 1
                all_right += 1
        
        # 结果排序，并和知识库结果求交集
        
        for doc,pro_list in doc_dic.items():
            sort_pre_tuple = sorted(pro_list,key=lambda d:d[0], reverse = True)
            sort_cate_len = len(sort_pre_tuple)
            pre_cate_list = []
            for pre_cate in sort_pre_tuple:
                pre_cate_list.append(pre_cate[1])
            if len(pre_cate_list) >=3:
                if label in pre_cate_list[:3]:
                    right += 1
                    all_right += 1
            elif label in pre_cate_list:
                right += 1
                all_right += 1
            # 和知识库结果求交集
            
            kb_pre_cate = []
            for cate in pre_cate_list:
                if cate in pro_list[0]:
                    kb_pre_cate.append(cate)
            kb_pre_len = len(kb_pre_cate)
            if kb_pre_len >=3:
                if cur_cate in kb_pre_cate[:3]:
                    right += 1
                    all_right += 1
            elif cur_cate in kb_pre_cate:
                right += 1
                all_right += 1
            
        acc = right/test_size
        print('acc %f'%acc)
        '''
        
        # if acc <= 0.4:
        #     save_file_lines(result_path+algorithm+'_less_0.2_way_0.txt',cur_cate+': '+str(acc)+'\n','a')
        test_one_end_time = time.time()
        one_run_time = round(test_one_end_time-test_one_time,4)
        print('test one cate time:%f\n'%one_run_time)

        save_file_lines(test_result_path,cur_cate+' dataset accuracy :%f'%acc+'\n','a')
        
    print('macro acc %f'%(all_right/all_len))
    test_end = time.time()
    test_run_time = round(test_end-test_start,4)
    print('merge test time: %f'%(test_run_time))

    save_file_lines(test_result_path,'using '+algorithm+' micro acc %f'%(all_right/all_len)+'\n','a')
    save_file_lines(test_result_path,'merge test time: %f'%(test_run_time),'a')
    