# -*- coding: utf-8 -*-
# @Author: bruce·li
# @Date:   2019-10-22 11:07:46
# @Last Modified by:   bruce·li
# @Last Modified time: 2019-10-22 17:38:32
# @Email:   talkwithh@163.com
import matplotlib
import matplotlib.pyplot as plt

y1=[40,106,176,250,318,394]
x1=[0.0,0.1,0.2,0.3,0.4,0.5]
y2=[40,106,176,250,318,394]
plt.plot(x1,y1,label='Category number of less than or equal to accuracy',linewidth=1,color='black',marker='o',markerfacecolor='black',markersize=5)
plt.plot(x1,y2,label='Category number of less than or equal to accuracy',linewidth=1,color='black',marker='o',markerfacecolor='black',markersize=5)

# plt.bar(x1,y1,label='Frist line',color='r')
for x,y in zip(x1,y1):
    plt.text(x, y+2, str(y), ha='right', va='bottom', fontsize=10,rotation=0)
for x,y in zip(x1,y2):
    plt.text(x, y+2, str(y), ha='right', va='bottom', fontsize=10,rotation=0)

plt.xlabel('Accuracy')
plt.ylabel('Category number')
# plt.title('')
plt.legend()
# plt.show() 

plt.savefig('cate_number.png', dpi=600)
