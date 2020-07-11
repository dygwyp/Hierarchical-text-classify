

from create_trible_dics import *
from write_lst_data_to_txt import *

def read_txt(fn):
    '''
    读取训练文件
    返回一个列表, 列表中的每个元素为文件的一行
    '''
    import os
    import codecs

    if os.path.isfile(fn):
        f_ = codecs.open(fn , 'r', 'utf-8')
        # str_ = f_.read()
        line_lst = f_.readlines()

        return line_lst

    else:
        print('没有输入正确的训练集文件')
        exit()


def get_CCN_of_corpus_lst(fn):
    '''
    chinese_category_number(中图分类号)
    '''

    line_lst = read_txt(fn)

    # 表头项 ['\ufeffid', '标题', '关键词', '摘要', '中图分类号']
    item_lst = line_lst[0].strip().split('\t')
    CCN_idx = item_lst.index('中图分类号')

    CCN_of_corpus_lst = []

    # 去除表头项
    for line in line_lst[1:]:
        lst = line.strip().split('\t')
        if len(lst) == 5:
            CCN_of_corpus_lst.append(lst[CCN_idx])
        else:
            print(line + ' Wrong!')

    return CCN_of_corpus_lst


def stats_multiple_classes_items(fn):
    CCN_of_corpus_lst = get_CCN_of_corpus_lst(fn)

    multiple_classes_items_num = 0
    class_num = 0
    
    for CCN_of_corpus in CCN_of_corpus_lst:
        if ';' in CCN_of_corpus:
            multiple_classes_items_num += 1
            class_num = CCN_of_corpus.count(';') + 1
        else:
            class_num += 1

    return multiple_classes_items_num, class_num



def get_stats_dic(dic, CCN_of_corpus_lst):

    # 将CCN_of_corpus_lst变成双列表
    # 增加一个未分类flag
    # 若flag == True, 则代表未分类
    # 若flag == False, 则代表已分类
    CCN_of_corpus_flag_lsts = []
    for CCN_of_corpus in CCN_of_corpus_lst:
        CCN_of_corpus_flag_lsts.append([CCN_of_corpus, True])


    # 三级统计字典
    stats_dic = dic.copy()

    for level_1 in dic.keys():
        print('get_stats_dic: level_1', level_1)

        for level_2 in dic[level_1].keys():
            # if level_2 != 'A1/49':
            #     continue
            # print('level_2', level_2)

            # 若dic[level_1][level_2]是空字典
            # 即二级类号没有三级类号的情况
            # 'A': 'A1/49': {}
            if not dic[level_1][level_2]:
                level_2_total = 0
                for idx, CCN_of_corpus_flag_lst in enumerate(CCN_of_corpus_flag_lsts): #原始数据分类号列表
                    # print('CCN_of_corpus_flag_lst', CCN_of_corpus_flag_lst)
                    
                    # 若该条记录已分类
                    # 则直接跳过
                    if CCN_of_corpus_flag_lst[1] == False:
                        continue

                    ccn = CCN_of_corpus_flag_lst[0]
                    if len(ccn):
                        if ';' not in ccn:
                            if level_2[0] == ccn[0] and level_2 in ccn:   #抽取语料库中存在的N级类目数据
                                # print(ccn + ' Match!!!')
                                level_2_total += 1
                                # 匹配之后, 才能设置为False
                                CCN_of_corpus_flag_lst[1] = False
                        # 类别数大于1时
                        else:        
                            ccn_lst = ccn.split(';')
                            
                            # 使用ccn_, 为了区分ccn
                            for ccn_ in ccn_lst:
                                # print('ccn', ccn)

                                if not len(ccn_):
                                    continue

                                # 类号的首字母相同 且 三级类号在语料的类号字符串中
                                elif level_2[0] == ccn_[0] and level_2 in ccn_:   #抽取语料库中存在的N级类目数据
                                    # print(ccn_ + ' Match!!!')
                                    level_2_total += 1

                                else:
                                    pass
                                    # print(ccn_ + ' pass!')

                    
                # 若二级类号没有三级类号
                # 则stats_dic[level_1][level_2]对应条目数, 而不是字典
                stats_dic[level_1][level_2] = level_2_total

            
            # dic[level_1][level_2]如果是空字典, 是不会执行以下循环的
            if type(dic[level_1][level_2]) == type({}):
                for level_3 in dic[level_1][level_2].keys():
                    # print('level_3', level_3)
                    # exit()
                    # if level_3 != 'A75':
                    #     continue

                    level_3_total = 0

                    for idx, CCN_of_corpus_flag_lst in enumerate(CCN_of_corpus_flag_lsts): #原始数据分类号列表
                        # print('CCN_of_corpus_flag_lst', CCN_of_corpus_flag_lst)
                        
                        # 若该条记录已分类
                        # 则直接跳过
                        if CCN_of_corpus_flag_lst[1] == False:
                            continue

                        ccn = CCN_of_corpus_flag_lst[0]
                        if len(ccn):
                            if ';' not in ccn:
                                if level_3[0] == ccn[0] and level_3 in ccn:   #抽取语料库中存在的N级类目数据
                                    # print(ccn + ' Match!!!')
                                    level_3_total += 1
                                    # 匹配之后, 才能设置为False
                                    CCN_of_corpus_flag_lst[1] = False
                            # 类别数大于1时
                            else:        
                                ccn_lst = ccn.split(';')
                                
                                # 使用ccn_, 为了区分ccn
                                for ccn_ in ccn_lst:
                                    # print('ccn', ccn)

                                    if not len(ccn_):
                                        continue

                                    # 类号的首字母相同 且 三级类号在语料的类号字符串中
                                    elif level_3[0] == ccn_[0] and level_3 in ccn_:   #抽取语料库中存在的N级类目数据
                                        # print(ccn_ + ' Match!!!')
                                        level_3_total += 1

                                    else:
                                        pass
                                        # print(ccn_ + ' pass!')
                    
                    stats_dic[level_1][level_2][level_3] = level_3_total
        
        # exit()
    return stats_dic


def stats_data_up_to_level_3(fn, category_fn, savefn):
    import os

    # 读取三级类目表
    dic = get_trible_dics(category_fn)

    # 统计信息存放在列表中
    stats_lst = []
    stats_level_1_lst = []
    stats_level_2_lst = []
    
    # 读取语料中的中图分类号
    # CCN: chinese_category_number
    CCN_of_corpus_lst = get_CCN_of_corpus_lst(fn)

    multiple_classes_items_num, class_num = stats_multiple_classes_items(fn)
    stats_sen = fn.rpartition('/')[-1] + '语料库中数据条数：'+ str(len(CCN_of_corpus_lst))
    stats_sen += '\n' + '语料库中有多标签的数据条数：'+ str(multiple_classes_items_num)
    stats_sen += '\n' + '语料库中标签的个数：'+ str(class_num)

    stats_lst.append(stats_sen)
    stats_level_1_lst.append(stats_sen)

    # 293421
    print(stats_sen)
    # exit()

    stats_dic = get_stats_dic(dic, CCN_of_corpus_lst)

    for level_1 in stats_dic.keys():
        level_1_total = 0

        for level_2 in stats_dic[level_1].keys():
            level_2_total = 0

            if not type(stats_dic[level_1][level_2]) == type({}):
                level_2_total += stats_dic[level_1][level_2]

            else:
                for level_3 in stats_dic[level_1][level_2].keys():
                    stats_lst.append(level_3 + '：' + str(stats_dic[level_1][level_2][level_3]))
                    level_2_total += stats_dic[level_1][level_2][level_3]

            # 对level_2进行统计
            stats_lst.append(level_2 + '：' + str(level_2_total))
            stats_level_2_lst.append(level_2 + '：' + str(level_2_total))

            level_1_total += level_2_total

        stats_lst.append(level_1 + '：' + str(level_1_total))
        stats_level_1_lst.append(level_1 + '：' + str(level_1_total))

    stats_total = 0

    # 从stats_level_1_lst的第1个元素开始统计
    # 第0个元素是'data1000_2_Linux.txt语料库中数据条数：293421\n...\n语料库中标签的个数：293421'
    for str_ in stats_level_1_lst[1:]:
        stats_total += int(str_.split('：')[-1])

    stats_level_1_lst.append('语料库中分类之后的统计数：'+ str(stats_total))
    print('语料库中分类之后的统计数：'+ str(stats_total))

    write_lst_data_to_txt(savefn, stats_lst)
    write_lst_data_to_txt(savefn.rpartition('/')[0] + '/' + savefn.rpartition('/')[-1].replace('.txt', '_level_1.txt'), stats_level_1_lst)
    write_lst_data_to_txt(savefn.rpartition('/')[0] + '/' + savefn.rpartition('/')[-1].replace('.txt', '_level_2.txt'), stats_level_2_lst)




def main():
    category_fn = '../2_ExtraData/category/classify_list.xlsx'
    fn = '../1_Data/unzip/data1000_2_Linux.txt'
    savepath = '../1_Data/stats_data_of_' + fn.rpartition('/')[-1] + '/'
    
    import os

    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    savefn = savepath + 'stats_data.txt'

    stats_data_up_to_level_3(fn, category_fn, savefn)


if __name__ == '__main__':
    main()