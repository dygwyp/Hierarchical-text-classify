
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


def get_text_lsts(text_fn):
    import jieba

    line_lst = read_txt(text_fn)

    text_lsts = []

    # 表头项 ['\ufeffid', '标题', '关键词', '摘要', '中图分类号']
    item_lst = line_lst[0].strip().split('\t')
    # 表头项项数
    item_num = len(item_lst)

    title_idx = item_lst.index('标题')
    keywords_idx = item_lst.index('关键词')
    abs_idx = item_lst.index('摘要')
    CCN_idx = item_lst.index('中图分类号')

    # 去除表头项
    for line in line_lst[1:5]:
        lst = line.strip().split('\t')
        if len(lst) == item_num:
            text = lst[title_idx] + ' ' + lst[keywords_idx] + ' ' + lst[abs_idx]
            text_lst = list(jieba.cut(text))

        text_lsts.append(text_lst)

    return text_lsts


def get_KB_dic(KB_fn):
    import codecs
    import json
    from collections import OrderedDict


    f = codecs.open(KB_fn, 'r', 'utf-8')
    # object_pairs_hook = OrderedDict并没有进行排序
    # KB_dic = json.load(f, object_pairs_hook = OrderedDict)
    KB_dic = json.load(f)
    # KB_dic = eval(f)

    return KB_dic



def get_level_3_from_KB(KB_dic, text_lsts):
    '''
    该函数
    参数:
    KB_dic: 以dic形式保存的KB(三级类目及其对应的关键词)
    text_lsts: 双列表, 每个单列表是分好词的字符串
    '''
    from collections import Counter

    level_3_score_lsts = []

    for text_lst in text_lsts:
        level_3_score_dic = {}
        for level_3 in KB_dic.keys():

            '''
            # 取交集计算得分, 方法太简单了
            level_3_KB_set = set(KB_dic[level_3].keys())
            text_set = set(text_lst)

            level_3_intersection_set = text_set.intersection(level_3_KB_set)
            score = len(level_3_intersection_set)
            '''

            # 取交集, 考虑出现的次数，计算得分
            level_3_KB_lst = list(KB_dic[level_3].keys())
            score = 0
            for level_3_KB in level_3_KB_lst:
                score += text_lst.count(level_3_KB)

            level_3_score_dic[level_3] = score
            
        c = Counter(level_3_score_dic)
        level_3_score_tup_lst = c.most_common()
        if len(level_3_score_tup_lst) > 20:
            level_3_score_tup_lst = level_3_score_tup_lst[:200]
        level_3_score_topN_dic = dict(level_3_score_tup_lst)
        level_3_score_topN_lst = list(level_3_score_topN_dic.keys())

        # print('level_3_score_topN_lst:', level_3_score_topN_lst)

        level_3_score_lsts.append(level_3_score_topN_lst)
        # exit()

    return level_3_score_lsts



def main():

    KB_fn = '../data/info/KB_json_of_data2'
    KB_dic = get_KB_dic(KB_fn)
    print('已读取KB_dic！')

    textfn = '../1_Data/unzip/data1000_2_Linux.txt'
    # text_lsts = get_text_lsts(textfn)
    # print(text_lsts)
    print('已读取text！')

    # 对于每个文本, 返回20个分类号
    text_lsts = [['发展','佛经','贡献','革新','资助','联手','合作','取材','哈同','独特','大藏经','主持','夫妇','顺应','救世','佛教']]
    level_3_score_lsts = get_level_3_from_KB(KB_dic, text_lsts)
    print(level_3_score_lsts)



if __name__ == '__main__':
    main()



