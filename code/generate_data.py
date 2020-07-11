# -*- coding: utf-8 -*-
"""
处理原始数据，生成训练数据集
@author: bruce
"""
import os,re
import time
import random
import jieba
import pandas as pd
from read_write_tool import read_file,save_file,read_file_lines

# 生成数据集
def generate_data(stopword_path,ori_datas,ch_file_path,target_path):
    '''
    ori_datas:原始语料，索引为id、标题、关键词、摘要、中图分类号、母体文献
    ch_file_path:层级目录文件
    stopword_path:停用词路径
    save_path:预处理后的文件保存路径
    '''
    start_time = time.time()
    print('='*40)
    #定义分类层级列表
    lev_n = ['1','2','3']
    #读取语料中的中图分类号
    ch_of_corpus = list(ori_datas['中图分类号'])
    print('..语料库中数据条数：'+str(len(ch_of_corpus)))
    #抽取中图分类号表中的二/三级ch号
    level2_ch = read_Ch_cate(ch_file_path,lev_n[1])
    print('..中图分类表中二级类目数：'+str(len(level2_ch)))
    #读取停用词表
    stopword = read_file_lines(stopword_path,'utf-8')

    # 抽取n级目录数据
    select_datas(ori_datas,ch_of_corpus,level2_ch,lev_n,target_path,stopword)
    
    # 统计数据集
    statistics_corpus(target_path)
    
    # 合并一级和三级类目数据
    fp_1 = open(target_path+'le1_data.txt','w',encoding='utf-8')
    fp_2 = open(target_path+'le1_data_w2v.txt','w',encoding='utf-8')
    fp_3 = open(target_path+'le2_data.txt','w',encoding='utf-8')
    merge_level3(target_path,fp_1,fp_2,fp_3)
    fp_1.close()
    fp_2.close()
    fp_3.close()

    end_time = time.time()
    run_slect_time = round(end_time-start_time,3)
    print('\n生成数据集运行时间：'+str(run_slect_time)+'s')

# 抽取数据
def select_datas(ori_datas,ch_of_corpus,level3_ch,lev_n,target_path,stopword):
    print('='*40)
    print('I-1.抽取数据,生成训练和测试数据集...')
    #三级类总数目统计
    all_cate_count = 0
    #总数据列表
    all_datas = []
    #一级类下三级类别数目统计
    le3_cate_count = 0
    #设置分类标志
    ch_of_corpus_flag = []
    for ccn in ch_of_corpus:
        if ';' in ccn:
            item_list = ccn.split(';')
            ch_of_corpus_flag.append([item_list,[True]*len(item_list)])
        else:
            ch_of_corpus_flag.append([ccn,[True]])
    #n级目录表在外循环，语料库分类号在内循环
    for ch in level3_ch:
        #一级类目下三级类别的数据条数统计
        le3_data_count = 0
        #一级类目下三级类别的数据list       # 可改为2级类
        le3_datas = []
        #三级类数据存放路径
        le3_datas_path = target_path + ch[0] +'/'
        if not os.path.exists(le3_datas_path):
            os.makedirs(le3_datas_path)
        for idx,ccn in enumerate(ch_of_corpus[:]): #原始数据分类号列表
            item_list = []
            if ';' in ccn:       #类别数大于1时
                item_list = ccn.split(';')
            else:
                item_list.append(ccn)
            for p in item_list:
                if not len(p):
                    continue
                if True not in ch_of_corpus_flag[idx][1]: #如果已抽取，将不再抽取
                    continue
                elif ch[0] == p[0] and ch in p: #抽取语料库中存在的N级类目数据
                    row = ori_datas.loc[idx]
                    title = row['标题']   #抽取数据标题
                    abstract = row['摘要']  #抽取数据摘要
                    key_word = row['关键词'] #抽取数据关键词
                    content = title+' '+abstract+' '+key_word
                    if len(content) < 10:
                        continue
                    #抽取后标记false
                    p_index = item_list.index(p)
                    ch_of_corpus_flag[idx][1][p_index] = False

                    con_seg = deal_data(content,stopword)
                    le3_datas.append('__label__'+ch+', '+' '.join(con_seg))
                    le3_data_count += 1

        if le3_data_count >= 500:
            #存储三级类数据
            save_file(le3_datas_path+ch[0]+'_data_count.txt',ch+'-->'+str(le3_data_count)+'\n','a')
            random.shuffle(le3_datas)
            write_data(le3_datas_path+ch[0]+'_data.txt',le3_datas)
            #打乱数据，使同类别的数据分散

# 读取中图分类号
def read_Ch_cate(le_file_path,le_n):
    infos = pd.read_excel(le_file_path)
    #抽取n级类号数据
    le_n_datas = infos.loc[infos['层级']==int(le_n)]
    le_n_id = list(le_n_datas['类号ID'])

    le_select = {}          #以字典形式存储分类号，可避免重复类号
    for i in le_n_id:
        if i not in le_select:
            le_select[i] = 0
    le_ccn_list = list(le_select.keys())
    le_ccn_list = sorted(le_ccn_list, reverse = True)
    return le_ccn_list

#读取数据
def read_data(ori_datas_path):
    # t = open(ori_datas_path,encoding='utf-8')
    con = pd.DataFrame(pd.read_csv(ori_datas_path,sep='\t'))
    con = con.fillna(value='')
    # print(con.iloc[47901])
    # print(len(con))
    return con

#预处理抽取的数据
def deal_data(content,stopword):
    seg_list = []
    #分词
    segs = jieba.lcut(content)               #分词
    segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
    segs=filter(lambda x:x not in stopword,segs)    #去掉停用词
    for i in segs:
        seg_list.append(i)
    un_list = list(set(seg_list))
    return un_list
#合并数据
def merge_level3(file_path,fp_1,fp_w2v,fp_3):
    print('='*40)
    print('I-3.合并数据...')
    file_list = []
    for root, dirs, files in os.walk(file_path, topdown=True):
        for name in files:
            if 'count' not in name and name[0].isupper():
                file_list.append(os.path.join(root, name))
                # print(os.path.join(root, name))
    for file in file_list:
        for con in open(file,'r',encoding='utf-8'):
            fp_3.write(con)
            # line_list = con.split(',')
            # label,text = line_list[0],line_list[1]
            
            # fp_w2v.write(text)
            # index = label.index('__',2)
            # le1_label = label[(index+2):(index+3)]
            # fp_1.write('__label__'+le1_label+','+text)  
        
#统计数据集
def statistics_corpus(data_path):
    print('='*40)
    print('I-2.统计数据...')
    level_dic = {}
    level1_count = 0
    all_doc_count = 0
    for level in os.listdir(data_path):
        if '.txt' in level or level.startswith('.'):
            continue
        temp_count = 0
        le1_list = os.listdir(data_path+level+'/')
        if len(le1_list):
            level_dic[level] = []
        else:
            os.rmdir(data_path+level+'/')
        for file in le1_list:
            if 'count' in file:
                con = read_file(data_path+level+'/'+file,'utf-8').split('\n')
                level1_count += len(con)-1
                level_dic[level].append(len(con)-1)
                for i in con[:-1]:
                    num = i.split('-->')[1]
                    temp_count += int(num)
                level_dic[level].append(temp_count)
                all_doc_count += temp_count
                print(str(len(con)-1)+'/'+str(temp_count))
    write_data(data_path+'statistics.txt','level-1\t二级类别数目\t文档数目\n','single')
    write_data(data_path+'statistics.txt',level_dic.items())
    write_data(data_path+'statistics.txt','\n总二级类别数目：%d'%level1_count+'\n总文档数目：%d'%all_doc_count,'single')
    # print(str(level_dic))
    # print(level1_count)

#写数据
def write_data(save_path,con,con_flag='list'):
    # print('保存fasttext格式数据...')
    with open(save_path,'a',encoding='utf-8') as fp:
        if con_flag == 'list':
            for i in con:
                fp.write(str(i)+'\n')
        else:
            fp.write(str(con))

if __name__ == '__main__':
    stopword_path = '../data/info/stopword/中文.txt'
    ori_path1 = '../data/original_data/ori_data/data1000_1.txt'
    ori_path2 = '../data/original_data/ori_data/data1000_2.txt'

    le_file_path = '../data/info/category/classify_list.xlsx'
    save_path = '../data/original_data/'

    print('1. 读取原始数据...')
    re = ''
    # con1 = read_data(ori_path1)
    # con2 = read_data(ori_path2)
    # re = con1.append(con2,ignore_index=True)

    main(stopword_path,re,le_file_path,save_path)
