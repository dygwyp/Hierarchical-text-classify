# create_double_dics.py


'''
读取一个txt文件
将文件内容生成双字典
例如
{entity:{alias:freq}}
'''

from write_lst_data_to_txt import *

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



def process_level_n_lst(level_n_lst):
    '''
    将level_n_lst中的诸如K3/7、B21/26修改为
    K3、K7和B21、B26
    '''
    level_n_lst_new = []

    for level_n in level_n_lst:
        if '/' in level_n:
            # 二级类号
            # K3/7
            # P1-093/1-097


            # 三级类号
            # B21/26
            # TH2/6
            i_list = level_n.split('/')

            # B21、TH2
            key_1 = i_list[0]
            if key_1 not in level_n_lst_new:
                level_n_lst_new.append(key_1)

            # 二级类号的T没有'/', 所以以下代码只会处理三级类号
            # TH2/6
            if level_n[0] == 'T':
                # TH6
                key_2 = i_list[0][:2] + i_list[1]
                if key_2 not in level_n_lst_new:
                    level_n_lst_new.append(key_2)

            # B21/26        
            else:
                # B26
                key_3 = i_list[0][0] + i_list[1]
                if key_3 not in level_n_lst_new:
                    level_n_lst_new.append(key_3)

        elif level_n not in level_n_lst_new:
            level_n_lst_new.append(level_n)
        
        else:
            pass

    return level_n_lst_new


def check_process_level_n_lst(fn):
    '''
    生成《level_3_lst.txt》和《level_3_lst_processed.txt》文件
    比较 针对带有'/'的类号字符串进行特殊处理之后 产生的影响
    '''
    level_1_lst, level_2_lst, level_3_lst = read_excel(fn)
    print(len(level_1_lst))         # 22
    print(len(level_2_lst))         # 244
    print(len(level_3_lst))         # 1754

    # write_lst_data_to_txt('level_3_lst.txt', level_3_lst)

    # # 二级和三级类号需要特殊处理
    # # 主要针对带有'/'的类号字符串进行特殊处理
    # level_2_lst = process_level_n_lst(level_2_lst)
    # level_3_lst = process_level_n_lst(level_3_lst)

    level_2_lst = rmvDup_lst(level_2_lst)
    level_3_lst = rmvDup_lst(level_3_lst)

    print(len(level_1_lst))         # 22
    print(len(level_2_lst))         # 244
    print(len(level_3_lst))         # 1679

    # write_lst_data_to_txt('level_3_lst_processed.txt', level_3_lst)



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

    # 三字典
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
        if len(level_3_belong2_level_1_lst) > len(level_3_lst_processed):
            # print(level_3_belong2_level_1_lst)
            # print(level_3_lst_processed)
            # exit()
            dic_NA = {}
            for level_3_belong2_level_1 in level_3_belong2_level_1_lst:
                if level_3_belong2_level_1 not in level_3_lst_processed:
                    dic_NA[level_3_belong2_level_1] = 1

            dic_2['NA'] = dic_NA


        dic[level_1] = dic_2

        
    return dic


def check_dic(fn):
    dic = get_trible_dics(fn)

    # 一级类号
    print(len(dic))                             # 22

    # 二级类号
    level_2_num = 0
    level_3_num = 0

    level_3_lst_from_dic = []
    for level_1 in dic.keys():
        level_2_num += len(dic[level_1])

        for level_2 in dic[level_1].keys():
            level_3_num += len(dic[level_1][level_2])

            for level_3 in dic[level_1][level_2].keys():
                level_3_lst_from_dic.append(level_3)

    print(level_2_num)                          # 250   比244多了6个‘NA’
    print(level_3_num)                          # 1679
    print(len(level_3_lst_from_dic))            # 1679
    # write_lst_data_to_txt('level_3_lst_from_dic.txt', level_3_lst_from_dic)



def main():
    # fn = '../2_ExtraData/category/classify_list.xlsx'
    fn = '../../data/info/category/classify_list.xlsx'
    # check_process_level_n_lst(fn)
    
    dic = get_trible_dics(fn)
    print(dic)
    
    # check_dic(fn)


if __name__ == '__main__':
    main()



