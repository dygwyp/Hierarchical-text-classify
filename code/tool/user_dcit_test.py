# -*- coding: utf-8 -*-
# @Author: bruce
# @Date:   2019-06-17 19:18:39
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-10-10 09:46:06
# @Email: talkwithh@163.com
'''
使用用户自定义词典进行分词
第一步：清洗知识组织核心词表,去掉单独一个字符长度的行
第二步：使用jieba工具进行分词。对比不使用自定义和使用自定义的差别
'''
import os
import jieba

def pre_process(con_list,kw_path):
    with open(kw_path,'w',encoding='utf-8') as fp_write:
        for line in con_list:
            if len(line) == 2:	# 加上'\n'
                continue
            fp_write.writelines(line)


def cut_word(test_path,user_dict_path):
    with open(test_path,'r',encoding='utf-8') as fp:
        data_list = fp.readlines()
        n=0
        for line in data_list:
            n+=1
            if n!=4:
                continue
            cut_re_bef = jieba.lcut(line)
            print(cut_re_bef)
            jieba.load_userdict(user_dict_path)
            cut_re_af = jieba.lcut(line)
            print(cut_re_af)
            # 基于目前的数据和词典，结果证明是有效的

def main():
    kw_path = '../data/info/dict/kw_dict.txt'
    kw_path_new = '../data/info/dict/kw_dict_new.txt'
    baike_dict = '../data/info/dict/clear_baike_word.txt'

    # 预处理

    # fp_read = open(kw_path,'r',encoding='utf-8')    
    # con_list = fp_read.readlines()
    # pre_process(con_list,kw_path_new)
    # fp_read.close()

    # 分词
    test_path = '../data/original_data/ori_data/test.txt'
    user_dict_path = kw_path
    
    cut_word(test_path,user_dict_path)


if __name__ == '__main__':
    main()