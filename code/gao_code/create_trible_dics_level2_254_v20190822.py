# create_double_dics.py


'''
读取一个txt文件
将文件内容生成双字典
例如
{entity:{alias:freq}}
'''

def read_excel(fn):
    import pandas as pd
    
    f = pd.read_excel(fn)

    level_1_data = f.loc[f['层级'] == 1]
    level_1_lst = list(level_1_data['类号ID'])

    level_2_data = f.loc[f['层级'] == 2]
    level_2_lst = list(level_2_data['类号ID'])

    level_3_data = f.loc[f['层级'] == 3]
    level_3_lst = list(level_3_data['类号ID'])


    return level_1_lst, level_2_lst, level_3_lst


def rmvNull(lst):
    '''
    去除列表里含有<null>的字符串
    '''

    temp = lst[:]    
    for str_ in temp:        
        if('<null>' in str_):   # 去除文本里的<null>
            lst.remove(str_)
           
    # print(lst)

    return lst

def rmvDup_lst(lst):
    '''
    去重
    '''

    temp = []
    m = 0
    
    while m < len(lst):
        if lst[m] not in temp:
            temp.append(lst[m])
        m += 1  


    return temp



def get_trible_dics(fn):
    '''
    {A:{A1:{A11:1, A12:1}},...,}
    '''
    import re
    
    level_1_lst, level_2_lst, level_3_lst = read_excel(fn)

    # 经过离线使用check_process_level_n_lst(fn)函数比较之后
    # 不再使用特殊处理
    # level_2_lst = process_level_n_lst(level_2_lst)
    # level_3_lst = process_level_n_lst(level_3_lst)

    level_2_lst = rmvDup_lst(level_2_lst)
    level_3_lst = rmvDup_lst(level_3_lst)

    # 三级字典
    dic = dict()
    for level_1 in level_1_lst:
        dic_2 = {}
        level_3_lst_processed = []
        level_3_belong2_level_1_lst = []

        # 要逆序循环level_2_lst
        # 否则会发生'A4': {'A491': 1}的情况 
        for level_2 in sorted(level_2_lst, reverse = True):
        # for level_2 in level_2_lst:

            if re.match(level_1, level_2):
                dic_3 = {}


                for level_3 in sorted(level_3_lst, reverse = True):
                    # 不能简单地使用 if level_2 in level_3: 来判断
                    # 否则会发生'B9': {'TB9': 1}的情况 
                    # if level_2 in level_3:

                    # 不能简单地使用 if re.match(level_2, level_3): 来判断
                    # 否则会发生'A4': {'A491': 1},  'A49': {'A491': 1} 的情况 
                    if re.match(level_2, level_3) and level_3 not in level_3_lst_processed:
                        dic_3[level_3] = 1
                        level_3_lst_processed.append(level_3)

                dic_2[level_2] = dic_3

        # 属于level_1的level_3的lst
        for level_3 in level_3_lst:
            if re.match(level_1, level_3):
                level_3_belong2_level_1_lst.append(level_3)

        # 若 属于level_1的level_3的个数 大于 之前处理过的level_3的个数
        # 则说明有些level_3并没有和level_2建立关系
        level_2_man_made_lst = []

        if len(level_3_belong2_level_1_lst) > len(level_3_lst_processed):
            # print(level_3_belong2_level_1_lst)
            # print(level_3_lst_processed)
            # exit()
            dic_NA = {}
            for level_3_belong2_level_1 in level_3_belong2_level_1_lst:
                if level_3_belong2_level_1 not in level_3_lst_processed:
                    
                    if level_1 == 'H':
                        # 虚构的二级类号
                        # H82_NA
                        level_2_man_made = level_3_belong2_level_1[:3] + '_NA'

                    else:
                        # 虚构的二级类号
                        # G5_NA、G6_NA等
                        level_2_man_made = level_3_belong2_level_1[:2] + '_NA'

                    if level_2_man_made not in level_2_man_made_lst:
                        dic_NA = {}
                        level_2_man_made_lst.append(level_2_man_made)

                    dic_NA[level_3_belong2_level_1] = 1
                    dic_2[level_2_man_made] = dic_NA

        # for le2 in level_2_man_made_lst:
        #     print('level_1 level_2_man_made dic_2[level_2_man_made]:', level_1, le2, dic_2[le2])

        dic[level_1] = dic_2

        
    return dic


def check_dic(dic, cate_lst):

    # 一级类号
    print('一级类号个数：%d' % len(dic))

    # 二级类号
    level_2_num = 0
    level_3_num = 0

    level_3_lst_from_dic = []
    for level_1 in dic.keys():
        level_2_num += len(dic[level_1])

        for level_2 in dic[level_1].keys():
            level_3_num += len(dic[level_1][level_2])

            for level_3 in dic[level_1][level_2]:
                level_3_lst_from_dic.append(level_3)

    print('二级类号个数：%d' % level_2_num)                          # 254   比244多了10个细化的‘NA’
    print('三级类号个数：%d' % level_3_num)                          # 840

    print('三级类号个数：%d' % len(level_3_lst_from_dic))            # 840  

    if len(cate_lst) > len(level_3_lst_from_dic):
        for cate in cate_lst:
            if cate not in level_3_lst_from_dic:
                print(cate)
