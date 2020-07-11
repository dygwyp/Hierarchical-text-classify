#coding='utf-8'
#读写文件的函数，包括bunch对象的读写
'''
@author: brunce.li
E-mail: 2471249566@qq.com
'''

import pickle
import codecs
#import chardet
from sklearn.externals import joblib

def save_file(savepath, content,way):
    # print('write file %s...'%(savepath))
    with codecs.open(savepath, way,encoding='utf-8') as fp:
        if type(content) is not str:
            fp.write(str(content))
        else:
            fp.write(content)

def save_file_lines(savepath,content,way='w'):
    # print('write file %s...'%(savepath))
    with codecs.open(savepath, way,encoding='utf-8') as fp:
        if type(content) is dict:       # 仅针对有new_test中文件
            for i,j in content.items():
                fp.write('\n'+i+'\n')
                try:
                    fp.write('le1 result: '+str(j[0][0])+'\n')
                    fp.write('le2 result:\n'+str(j[0][1])+'\n')
                    fp.write('final result:\n'+str(j[0][2])+'\n')
                except:
                    fp.write('null'+'\n')
        else:
            for con in content:
                fp.write(str(con)+'\n')


def read_file(file_name,encode='utf-8'):
    print('read file %s...'%(file_name))
    if '.txt' not in file_name:
        return 0
    with open(file_name, "r",encoding=encode) as fp:
        content = fp.read()
    return content

# read file
# return param:list
def read_file_lines(file_name,encode='utf-8'):
    '''
    #判断文件编码类型
    with codecs.open(path,'rb') as fp:
        con = fp.read()
        result = chardet.detect(con)
    '''
    # print('read file %s...'%(file_name))
    if '.txt' not in file_name:
        return 0
    with codecs.open(file_name, "r+",encode) as fp:
        content = fp.readlines()
    return content

def writebunch_obj(path, bunchobj):
    with codecs.open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

def readbunch_obj(path):
    with codecs.open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

def save_model(model,model_path):
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)
