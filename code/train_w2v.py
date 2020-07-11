# -*- coding: utf-8 -*-
# @Author: bruce
# @Date:   2019-05-15 10:05:17
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-10-10 10:10:20
# @Email: talkwithh@163.com
'''
训练词向量
'''
import time
import logging
import os,json
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from read_write_tool import read_file

# 训练词向量
def train_word2vec(train_path,model_save_path):
	data = read_file(train_path,'utf-8').split('\n')
	start = time.time()
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = Word2Vec(data, sg=0,hs=1,size=200,window=5, min_count=5, iter=50)    #右边是：sg=1;;左边是：sg=0
	model.save(model_save_path+'w2v_iter_20_model')
	end = time.time()

	print('训练时间%s'%str(start-end))
    
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
# 得到词向量模型的词语
def get_w2v_word(w2v_model):
    word_voc = w2v_model.vocab
    if type(word_voc) is dict:
        word_dic = word_voc.keys()
        word_list = list(word_dic)

    return word_list

# 去除停用词
def deal_word_list(word_list_path,stopword_path):
    new_word_list = []
    word_list = read_file_lines(word_list_path)
    stopword_list = read_file_lines(stopword_path)
    new_stopword_list = []
    for sw in stopword_list:
        sw = sw.strip()
        new_stopword_list.append(sw)
    for word in word_list:
        word = word.strip()
        if word not in new_stopword_list:
            new_word_list.append(word)

    return new_word_list

def save_list(word_list,save_path):
    with open(save_path,'w',encoding='utf-8') as fp:
        for word in word_list:
            fp.write(word+'\n')
            
def test_model(sent,w2v_model):
    # dic_list = w2v_model['传统']
    # print(dic_list)
    train_w2v = get_sent_vec(300,sent,w2v_model)

#compute the vector of each sentence
def get_sent_vec(size,sent,model,skip_word_save_path=''):
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
    if skip_word_save_path != '':
        with open(skip_word_save_path+'skip_word_baidu_test.txt','a',encoding='utf-8') as fp:
            if skip_count:
                for word in skip_word_list:
                    fp.write(word+',')
                fp.write('\tword count:{},'.format(count))
                fp.write('skip count:{}\n'.format(skip_count))
    # print('word count:{}'.format(count))
    # print('skip count:{}'.format(skip_count))
    # print('rate:{}\n'.format(skip_count/count))
    return vec

#get train vector
def get_train_vec(x_train,w2v_model,skip_word_save_path=''):
    
    train_vec = np.concatenate([get_sent_vec(300,sent,w2v_model,skip_word_save_path) for sent in x_train])
    # test_vec = np.concatenate([get_sent_vec(300,sent,w2v_model) for sent in x_test])
    #保存数据
    # np.save(doc_vec_path+'train_vec.npy',train_vec)
    # np.save('E:/NLP/chinese-w2v-sentiment/test_vec.npy',test_vec)
    return train_vec

def clear_w2v(w2v_model_path):
    from tqdm import tqdm
    n = 0
    with open(w2v_model_path+'new_tenxun_w2v.txt', 'a',encoding='utf-8', errors='ignore') as w_f:
        with open(w2v_model_path+'tenxun_word2vec.txt', 'r',encoding='utf-8', errors='ignore')as f:
            for i in tqdm(range(8824330)): #似乎不同时期下载的词向量range并不一样
                data = f.readline()
                a = data.split()
                if i == 0:
                    w_f.write('8748463 200\n') #行数的写入也有可能不同
                if len(a) == 201:
                    if not a[0].isdigit():
                        n=n+1
                        w_f.write(data)
    # print(n)  #输出清洗后的range

if __name__ == '__main__':
    data_path = '../data/original_data/target_path/le1_data_w2v.txt'
    w2v_model_path = '../data/model/W2V/sgns.baidubaike.bigram-char'
    sent = ['传统','说明','圣经','述评','推至','宽广','批判','代表','重要','产物','历经','产生']
    w2v_model = load_w2v_model(w2v_model_path)
    test_model(sent,w2v_model)
	# train_word2vec(data_path,w2v_model_path)