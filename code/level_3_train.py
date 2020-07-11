# -*- coding: utf-8 -*-
# @Author: bruce
# @Date:   2019-05-15 15:44:17
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-07-14 14:40:51
# @Email: talkwithh@163.com
'''
训练分类器
'''
import os,time
import random
import json
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from algorithm_model import SVM_model,bayes_model,sgd_model
from pre_process import get_dataset
# from json_encoder import MyEncoder
from train_w2v import get_train_vec,load_w2v_model
from read_write_tool import read_file,save_file,read_file_lines,save_file_lines,load_model,save_model

NUMBER = 10  #每个类训练的二分类器数目
RANGE_SIZE_OF_DATA = [500,1500]  # 每个类抽取数据条数范围
SCALE_OF_DATA = [0.5,0.4,0.1]   # 二分类训练集、融合训练集和测试集比例

# 训练二分分类器
def train_binary_classifier(algorithm,train_path,model_save_path,result_save_path):
	# 获取类别列表
    # 加载数据
    train_dic = json.loads(read_file(train_path))  
    cate_list = list(train_dic.keys())
    class_number = len(cate_list)
    train_start = time.time()
    
    result_save_full_path = result_save_path+'binary_result/'
    if not os.path.exists(result_save_full_path):
        os.makedirs(result_save_full_path)

    # 加载模型
    clf = get_model(algorithm)

    print('..训练分类器')
    all_score = 0.0
    results,less_score_cate = '',''
    model_name_map = {}	#训练文本映射字典
    for cate in cate_list:
        sum_score = 0.0
        if not os.path.exists(model_save_path+cate+'/'):
            os.makedirs(model_save_path+cate+'/')
        for i in range(NUMBER):
            train,label = train_dic[cate][i][0],train_dic[cate][i][1]
            x_train,x_test,y_train,y_test = train_test_split(train,label,test_size=0.2)

            classifier = clf.fit(x_train,y_train)
            pred = clf.predict(x_test)
            score = metrics.accuracy_score(y_test, pred)
            sum_score += score
            # save_model
            model_full_path = model_save_path+cate+'/'+algorithm+'_'+cate+'_'+str(i+1)+'.model'
            save_model(classifier,model_full_path)
            # model_name_map[model_full_path] = train_dic[cate][i][2]
        avg_score = sum_score/NUMBER
        all_score += sum_score
        if avg_score <= 0.85:
            save_file_lines(result_save_full_path+algorithm+'_score_less0.85.txt',str(cate)+'\n','a')
        result_str = "%s avg-accuracy:   %0.3f " % (cate,score)
        print(result_str)
        results = result_str+'\n'
        save_file_lines(result_save_full_path+algorithm+'_test.txt',results,'a') 
        # break
    all_avg = all_score/(NUMBER*class_number)
    all_result_str = "%s all avg-accuracy:   %0.3f \n" % (algorithm,all_avg)
    print(all_result_str)

    train_end = time.time()
    train_run_time = round(train_end-train_start,4)
    run_time_str = algorithm+' train time: %f'%(train_run_time)
    print(run_time_str)
    
    save_file_lines(result_save_full_path+algorithm+'_test.txt',all_result_str+'\n'+run_time_str,'a')
    # save_file(result_save_path+algorithm+'_model_name_map.txt',str(model_name_map),'w')
    # model_name_json = json.dumps(model_name_map,cls=MyEncoder)      # 速度比上一行快
    # save_file(result_save_path+algorithm+'_model_name_map_json.txt',model_name_json,'w')

def train_merge_classifier(algorithm,train_merge_path,w2v_model,model_save_path,result_save_path):
    # generate merge sub-dataset
    print('..生成融合训练集')
    train_start = time.time()
    train_merge_data = eval(read_file(train_merge_path))
    cate_list = list(train_merge_data.keys())
    class_number = len(cate_list)
    # 得到融合训练集
    train_merge_dic = get_merge_dataset(class_number,cate_list,train_merge_data)
    # 结果保存路径
    merge_result_save_path = result_save_path + 'merge_result/'
    if not os.path.exists(merge_result_save_path):
        os.makedirs(merge_result_save_path)

    # 融合模型保存路径
    merge_model_path = model_save_path +'merge_model/'
    if not os.path.exists(merge_model_path):
        os.makedirs(merge_model_path)

    # train merge classifier
    # 加载二分类器映射文件
    # model_name_map = eval(read_file(result_save_path+algorithm+'_model_name_map.txt'))
    # model_name_map = json.loads(read_file(result_save_path+algorithm+'_model_name_map_json.txt'))   # 速度比上一行快

    # get sklearn classifier model
    clf = get_model(algorithm)
    # 加载w2v模型
    # w2v_model = load_w2v_model(w2v_model_path)
    # load binary models
    model_dic = load_binary_model(algorithm,class_number,cate_list,model_save_path)

    print('..训练')
    sum_score = 0.0
    all_less_str,all_result_str = '',''
    for k in range(class_number):
        start_time = time.time()
        cur_cate = cate_list[k]
        item = train_merge_dic[cur_cate][0]
        # print(train_merge_dic[cur_cate][0])   [['text1','text2'],['R','-R']]
        con,labels = item[0],item[1]
        merge_size = len(labels)
        
        lb_list = []
        text_pro = []
        for label in labels:
            if label == cur_cate:
                lb_list.append(1)
            else:
                lb_list.append(0)
        pro_matrix = np.array([],[])
        for model in model_dic[cur_cate]:
            # clf = load_model(model)
            binary_model = model[1]
            # voc = model_name_map[model[0]]
            train_w2v = get_train_vec(con,w2v_model)
            # vectorizer = TfidfVectorizer(vocabulary=voc)
            # tdm = vectorizer.fit_transform(con)
            pred = binary_model.predict_proba(train_w2v)   # pred = clf.predict(tdm)
            for i in range(len(pred)):
                text_pro.append(pred[i][1])
        pro_matrix = np.array(text_pro).reshape((NUMBER,merge_size)).T
        '''
        lb_list = []
        all_text_pro = []
        for text,label in zip(con,labels):
            text_pro = []
            if label == cur_cate:
                lb_list.append(1)
            else:
                lb_list.append(0)
            for model in model_dic[cur_cate]:
                print(model)
                print(text)
                clf = load_model(model)
                voc = model_name_map[model]
                vectorizer = TfidfVectorizer(vocabulary=voc)
                tdm = vectorizer.fit_transform([text])
                pred = clf.predict_proba(tdm)   # pred = clf.predict(tdm)
                print(pred)
                text_pro.append(pred[0][1])
                print(pred[0][1])
                break
            all_text_pro.append(text_pro)
            # print(text_pro)
            break

        # print(len(all_text_pro))
        # print(all_text_pro[:5])
        '''
        # training
        x_train,x_test,y_train,y_test = train_test_split(pro_matrix,lb_list,test_size=0.3)
        classifier = clf.fit(pro_matrix,lb_list)
        pred = clf.predict(x_test)
        score = metrics.accuracy_score(y_test, pred)
        if score <= 0.85:
            all_less_str += cur_cate+':'+str(score)+'\n'
            
        sum_score += score
        result_str = '%s merge classifier accuracy : %f\n'%(cur_cate,round(score,3))
        all_result_str += result_str
        print(result_str)
        save_file_lines(merge_result_save_path+algorithm+'_test.txt',result_str,'a')
        model_full_path = merge_model_path + cur_cate + '_'+algorithm+'_merge.model'
        save_model(classifier,model_full_path)

        end_time = time.time()
        print('one run time {}\n'.format(end_time-start_time))

    avg_score = sum_score/class_number
    avg_score_str = algorithm+' merge classifier avg accuracy %f'%avg_score
    print(avg_score_str)

    train_end = time.time()
    train_run_time = round(train_end-train_start,4)
    merge_run_time_str = 'merge train time: %f'%(train_run_time)
    print(merge_run_time_str)

    save_file_lines(merge_result_save_path+algorithm+'_score_less0.85.txt',all_less_str,'w')
    # save_file_lines(merge_result_save_path+algorithm+'_test.txt',all_result_str,'w')
    save_file_lines(merge_result_save_path+algorithm+'_test.txt',avg_score_str+'\n'+merge_run_time_str,'a')

def get_merge_dataset(class_number,cate_list,train_merge_data):
    train_merge_dic = {}
    for i in range(class_number):
        cur_cate = cate_list[i]
        train_merge_dic[cur_cate] = []
        self_data = train_merge_data[cur_cate]
        self_data_length = len(self_data)
        merge_size = int(SCALE_OF_DATA[1]*self_data_length)

        other_data = [] # 其它数据初始化
        for j in range(NUMBER):
            other_cate = cate_list[(j*5+class_number//NUMBER)%class_number]  # 获取其他类的融合训练数据
            other_data_list = train_merge_data[other_cate]

            if other_cate[0] == cur_cate[0]:
                continue
            elif len(other_data) > len(self_data):
                break
            else:
                other_data += random.sample(other_data_list, int(merge_size//NUMBER))
        cur_con = self_data+other_data
        self_con,self_label = get_dataset(cur_con,cur_cate)
        train_merge_dic[cur_cate].append([self_con,self_label])

    return train_merge_dic

# 加载分类器接口
def get_model(algorithm):
    if algorithm == 'sgd':
        clf = sgd_model()
    elif algorithm == 'svm':
        clf = SVM_model()
    elif algorithm == 'bayes':
        clf = bayes_model()

    return clf

# 加载二分类模型
def load_binary_model(algorithm,class_number,cate_list,model_save_path):
    model_dic = {}
    for j in range(class_number):
        cur_cate = cate_list[j]
        model_dic[cur_cate] = []
        model_path = model_save_path+cur_cate+'/'
        models = os.listdir(model_path)
        for model in models:
            if algorithm in model:      # sgd,svm      sgd效果更好
                model_full_path = model_path + model
                model_dic[cur_cate].append([model_full_path,load_model(model_full_path)])

    return model_dic
