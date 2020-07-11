# -*- coding: utf-8 -*-
# @Author: bruce·li
# @Date:   2019-07-08 10:02:28
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-07-14 16:18:48
# @Email:   talkwithh@163.com
'''
分类接口
输入：未标注数据
输出：每条记录标注三个3级中图分类号
备注：标注结果在输入文件的每一行的末尾
'''
import os,time
import codecs
import jieba
import numpy as np
import fastText.FastText as ff  #win系统
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from read_write_tool import read_file,save_file,read_file_lines,save_file_lines,load_model,save_model

NUMBER = 10  #每个类训练的二分类器数目
SCALE_OF_DATA = [0.5,0.4,0.1]   # 二分类训练集、融合训练集和测试集比例

# 加载jieba自定义词典
def load_user_dict(userdict_path):
    start_time = time.time()
    jieba.load_userdict(userdict_path)

    end_time = time.time()
    print('加载自定义词典用时：{}s'.format(end_time-start_time))

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

# 加载融合模型
def load_merge_model(algorithm,model_save_path):
    model_merge_dic = {}
    for model in os.listdir(model_save_path):       
        if algorithm in model:
            le3_cate = model.split('_')[0]
            model_merge_dic[le3_cate] = load_model(model_save_path+model)

    return model_merge_dic

# 加载词向量模型
def load_w2v_model(model_save_path):
    start_time = time.time()
    if model_save_path.endswith('model'):
        model = Word2Vec.load(model_save_path)
    elif model_save_path.endswith('bin'):
        model = KeyedVectors.load_word2vec_format(model_save_path, binary=True)
    else:
        model = KeyedVectors.load_word2vec_format(model_save_path, binary=False)
    end_time = time.time()
    print('load {} model time: {}s'.format(model_save_path,end_time-start_time))

    return model

# 加载fasttext模型
def load_fasttext(model_path):
    classifier = ff.load_model(model_path)
    return classifier

# fasttext test
def fasttext_test(input_text,classifier):
    result = classifier.predict(input_text)
    # print('预测一级类别：%s'%(result[0]))
    return result[0]

#get train vector
def get_train_vec(x_train,w2v_model):
    train_vec = np.concatenate([get_sent_vec(300,sent,w2v_model) for sent in x_train])

    return train_vec

#compute the vector of each sentence
def get_sent_vec(size,sent,model):
    vec = np.zeros(size).reshape(1,size)
    count = 0
    skip_count = 0
    skip_word_list = []
    if type(sent) is str:
        sent_list = sent.split()

    for word in sent_list:
        try:
            vec += model[word].reshape(1,size)
            count += 1
        except:
            skip_count += 1
            skip_word_list.append(word)
            continue
    if count != 0:
        vec /= count
    # print('word count:{}'.format(count))
    # print('skip count:{}'.format(skip_count))
    # print('rate:{}\n'.format(skip_count/count))
    return vec
def ccn_style(f, lsts_data):
    '''
    lsts_data = [['你好，...','<p>《你好，中国》...。</p>'], ...,[]]
    lst_data = ['你好，...','<p>《你好，中国》...。</p>']
    
    '''
    sen = ''
    for idx, lst_data in enumerate(lsts_data):
        sen = ''
        # check时使用
        # sen = str(idx + 1) + '\t'
        
        for str_ in lst_data:
            sen = sen + str_ + ';'

        sen = sen.strip() + '\n'
        f.write(sen)


def write_lsts_data_to_txt(savefn, lsts_data):
    '''
    把抽取出来的信息写入txt
    '''
    if os.path.isfile(savefn):
        print(savefn + ' already exists!')
    else:
        f = codecs.open(savefn , 'w' , 'utf-8')    #创建txt文件


        if '_' in savefn:
            ccn_style(f, lsts_data)
        else:
            print("There's no style for the txt file!")

        f.close()

def load_pre_model(algorithm_name, userdict_fn,le1_fasttext_model_path,le2_fasttext_model_path,w2v_model_path,binary_model_path):
    # load model
    print('*'*30+'加载模型'+'*'*30)
    load_model_time = time.time()

    # 加载jieba自定义词典
    jieba.load_userdict(userdict_fn)

    # load fasttext model
    le1_fasttext_model = load_fasttext(le1_fasttext_model_path)
    print('加载 一级分类器模型 已完成！')
    le2_fasttext_model = load_fasttext(le2_fasttext_model_path)
    print('加载 二级分类器模型 已完成！')

    # 加载词向量模型
    print('开始加载 词向量模型！大约需要5 Mins 请耐心等待！')
    w2v_model = load_w2v_model(w2v_model_path)
    print('加载 词向量模型 已完成！')
    
    # 类目统计信息
    categories_list = os.listdir(binary_model_path)
    cate_list = [cate for cate in categories_list if 'merge' not in cate]
    merge_model_path = binary_model_path + 'merge_model/'
    class_number = len(cate_list)
    
    # 加载二分类模型
    model_dic = load_binary_model(algorithm_name,class_number,cate_list,binary_model_path)
    print('加载 二分类模型 已完成！')

    # 加载融合模型
    model_merge_dic = load_merge_model(algorithm_name,merge_model_path)
    print('加载 融合模型 已完成！')
    print('三级类类目数目：{}'.format(class_number))

    load_end_time = time.time()
    print('加载模型用时：{}'.format(load_end_time-load_model_time))

    return le1_fasttext_model, le2_fasttext_model, w2v_model, model_dic, model_merge_dic


def get_stopwords_lst(stopwords_fn):
    line_lst = read_file_lines(stopwords_fn)

    stopwords_lst = [line.strip() for line in line_lst]

    return stopwords_lst

# 预处理
def pre_process(input_fn,stopwords_fn):
    # 数据预处理
    print('*'*30+'数据预处理'+'*'*30)
    preprocess_start_time = time.time()

    line_list = read_file_lines(input_fn)
    stopwords_lst = get_stopwords_lst(stopwords_fn)
    pre_result_list = []

    for idx, line in enumerate(line_list[1:]):
        if idx % 1000 == 0:
            print('No.' + str(idx) + ' starts to be preprocessed')

        line_list = line.split('\t')
        # 最后两列是 原文分类号 和 机标分类号
        # 前两列是 序号 和 摘要
        content = ' '.join(line_list[2:])
       
        # seg_list = []
        segs = jieba.lcut(content)             #分词
        seg_list = list(segs)

        # 速度较慢
        # segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
        # segs=filter(lambda x:x not in stopword,segs)    #去掉停用词
        # 优化代码
        cws_list = [seg for seg in seg_list if len(seg) > 1 and seg not in stopwords_lst ]
        pre_result_list.append(' '.join(cws_list))

    print('数据总数：{}'.format(len(pre_result_list)))
    # print(pre_result_list[:5])
    preprocess_end_time = time.time()
    print('数据预处理用时：{}'.format(preprocess_end_time-preprocess_start_time))

    return pre_result_list

def predict_classification(le1_fasttext_model, le2_fasttext_model, w2v_model, model_dic, model_merge_dic, test_data_list):    
    print('*'*30+'预测类别'+'*'*30)
    predict_start_time = time.time()
    # 一级类目预测/二级
    level_1_pre_result = fasttext_test(test_data_list,le1_fasttext_model)
    level_2_pre_result = fasttext_test(test_data_list,le2_fasttext_model)
    level_1_pre_labels_list,level_2_pre_labels_list = [],[]
    for le1 in level_1_pre_result:
        label_le1_list = le1[:-1].split('__')
        level_1_pre_labels_list.append(label_le1_list[-1])

    for le2 in level_2_pre_result:
        label_le2_list = le2[:-1].split('__')
        level_2_pre_labels_list.append(label_le2_list[-1])

    text_pre_results = {}
    pre_result_dict,final_result_dict = {},{}

    text_flag_md_dict = {}
    for i,text in enumerate(test_data_list):
        if i%100 == 0:
            print('No ' + str(i) + ' starts to be predicted')
        le_flag = 0
        text_flag_md_dict[i] = []
        text_pre_results[str(i)+'\t'+text] = []
        text_le1_label,text_le2_label = level_1_pre_labels_list[i],level_2_pre_labels_list[i]
        text_pre_results[str(i)+'\t'+text].append([text_le1_label,text_le2_label])

        if text_le2_label[0] == text_le1_label:
            le_flag = True
        for bin_cate,models in model_dic.items():
            pre_result_dict[bin_cate] = []
            skip_flag = 0
            if not le_flag and bin_cate[0] in text_le1_label:
                skip_flag = 1
            if le_flag and bin_cate[:2] in text_le2_label:
                skip_flag = 2
            if skip_flag:
                text_flag_md_dict[i].append(bin_cate)

    print('批量预测数据...')
    test_size = len(test_data_list)
    doc_dic = {i:[] for i in range(test_size)}
    for bin_cate,models in model_dic.items():
        text_pro = []
        pro_matrix = np.array([],[])
        for model in models:
            # clf = load_model(model)
            clf = model[1]
            train_w2v = get_train_vec(test_data_list,w2v_model)
            pred = clf.predict_proba(train_w2v)
            for i in range(len(pred)):
                text_pro.append(pred[i][1])
        pro_matrix = np.array(text_pro).reshape((NUMBER,test_size)).T
        temp_pro = []
        for c,md in model_merge_dic.items():
            if bin_cate in c:
                # print('load merge model %s'%mer_md)
                merge_model = md
                pre = merge_model.predict_proba(pro_matrix)
                for j in range(len(pre)):
                    doc_dic[j].append([pre[j][1],c])

    print('筛选结果...')
    for i,pro_list in doc_dic.items():
        sort_pre_tuple = sorted(pro_list,key=lambda d:d[0], reverse = True)
        sort_cate_len = len(sort_pre_tuple)
        pre_cate_list = []
        final_result_dict[i] = []
        for pre_cate in sort_pre_tuple:
            if pre_cate[1] in text_flag_md_dict[i]:
                pre_cate_list.append(pre_cate[1])

        pre_len = len(pre_cate_list)
        if pre_len >=3:
            final_result_dict[i] = pre_cate_list[:3]
        else:
            final_result_dict[i] = pre_cate_list 

        text = test_data_list[i]
        text_pre_results[str(i)+'\t'+text][0].append(final_result_dict[i])

    # print(final_result_dict)
    final_result_lsts = list(final_result_dict.values())

    # 预测时间
    predict_one_time = time.time()
    one_run_time = round(predict_one_time-predict_start_time, 4)
    print('predict time:%f\n' % one_run_time)

    return final_result_lsts, text_pre_results

def save_result_into_files(algorithm_name, way_name, input_fn, final_result_lsts, text_pre_results):
    # 保存结果
    result_path = input_fn.rpartition('.')[0] +'_result'+'/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # save_file_lines(input_fn+'result.txt',final_result_lsts)
    savefn = result_path + algorithm_name + way_name + '_result_2.txt'
    write_lsts_data_to_txt(savefn, final_result_lsts)

    # 保存中间结果
    savefn_of_mid_result = result_path + algorithm_name + way_name + '_mid_result_2.txt'
    save_file_lines(savefn_of_mid_result, text_pre_results, 'w')

def main():
    # 停用词路径
    stopword_path = '../data/info/stopword/中文.txt'
    # jieba分词自定义词典
    userdict_path = '../data/info/dict/Baidu_lexicon.txt'

    # 模型路径
    base_model_path = '../data/model/'
    # level-1和level-2路径
    le1_fasttext_model_path = base_model_path+'level_1/level_1_fasttext_classifier_big_big.model'
    le2_fasttext_model_path = base_model_path+'level_2/level_2_fasttext_classifier_big_big.model'
    # 二分类模型路径
    binary_model_path = base_model_path+'level_3_big_big/'
    # 词向量模型
    w2v_model_path = base_model_path+'W2V/sgns.baidubaike.bigram-char'

    # 加载模型
    algorithm_name = 'svm'
    le1_fasttext_model, le2_fasttext_model, w2v_model, model_dic, model_merge_dic = load_pre_model(algorithm_name, userdict_path,le1_fasttext_model_path,le2_fasttext_model_path,w2v_model_path,binary_model_path)

    # 测试路径
    input_file_path = '../data/test_data/中宏产业报告.txt'
    # 分类和标注
    way_name = 'way_le1_le2_w2v'
    line_list = pre_process(input_file_path,stopword_path)
    final_result_lsts, text_pre_results = predict_classification(le1_fasttext_model, le2_fasttext_model, w2v_model, model_dic, model_merge_dic, line_list)
    save_result_into_files(algorithm_name, way_name, input_file_path, final_result_lsts, text_pre_results)
    # 分类和标注测试

if __name__ == '__main__':
    main()
