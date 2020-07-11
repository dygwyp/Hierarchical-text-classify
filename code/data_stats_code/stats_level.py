# -*- coding: utf-8 -*-
# @Author: bruce·li
# @Date:   2019-10-18 10:59:54
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-10-19 14:57:27
# @Email:   talkwithh@163.com
'''
获取数据集的二级类、三级类数据信息。
数据输入1：
输入路径：/data/original_data/target_path/
输入文件：输入路径+A/A_data_count.txt
输入文件内容如：A849-->1526\nA81-->1001
数据输入2：
输入文件：/data/info/category/classify_list.xlsx

输出：
数据集二级类目和三级类目数量和文档数量信息
'''
import os
import pandas as pd

def main(dataset_path,le_file_path):
    # 初始化列表、字典
    level2_list = []
    level2_dic = {}
    level3_dic = {}

    # 得到数据集信息文件列表
    data_info_list = dataset_info(dataset_path)
    # print(data_info_list)
    # 得到中图分类二级分类列表
    le2_ccn_list = read_Ch_cate(le_file_path,'2')
    # print(le2_ccn_list)

    for data_info in data_info_list:
        info_line_list = read_file_lines(data_info)
        for line in info_line_list:
            if line[0] not in level2_dic:
                level2_dic[line[0]] = 0
            i = 2 
            while line[:i] in le2_ccn_list:
                if line[:i] not in level2_list:
                    level2_list.append(line[:i])
                    level2_dic[line[0]] += 1
                if i == 3:
                    break
                i += 1

    print(level2_list)
    # print(len(level2_list))
    print(level2_dic)

    return level2_list

# 返回数据集信息文件列表
def dataset_info(dataset_path):
    data_info_list = []
    for root,dirs,files in os.walk(dataset_path):
        for file in files:
            if 'count' in file:
                data_info_list.append(os.path.join(root,file))
    return data_info_list

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

# read file
def read_file_lines(file_name):
    with open(file_name, "r",encoding='utf-8') as fp:
        content = fp.readlines()
    return content

if __name__ == '__main__':
    dataset_path = '../../data/original_data/target_path/'
    le_file_path = '../../data/info/category/classify_list.xlsx'
    main(dataset_path,le_file_path)