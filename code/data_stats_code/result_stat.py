# -*- coding: utf-8 -*-

'''
统计三级类分类效果
'''
import os
from read_write_tool import read_file,save_file,read_file_lines

def stat_le3_acc(le1_path,result_path,save_path,threshold):
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	le1_list = read_file_lines(le1_path)
	results = read_file_lines(result_path)
	save_str = ''
	count = 0
	le3_acc_result_dic = {i.strip():0 for i in le1_list}
	print(le3_acc_result_dic)
	for line in results[:-2]:
		line_list = line.split(':')
		cate = line_list[0].split()[0]
		acc = line_list[1].strip()
		
		if float(acc) <= threshold:
			save_str += cate+':'+str(acc)+'\n'
			le3_acc_result_dic[cate[0]] += 1
			count += 1
	print('less '+str(threshold)+' category counts:'+str(count)+'\n')
	# save_file(save_path+'less_'+str(threshold)+'.txt',save_str,'w')
	# save_file(save_path+'le3_acc_result.txt','less '+str(threshold)+' category counts:'+str(count)+'\n','a')
	save_file(save_path+'le3_acc_distribute_result.txt','less '+str(threshold)+':\n'+str(le3_acc_result_dic)+'\n','a')

def stat_le1_error(records_path,le1_save_path):
	if not os.path.exists(le1_save_path):
		os.makedirs(le1_save_path)
	le1_error_dic = {}	
	records_file_list = os.listdir(records_path)
	for file in records_file_list:
		if not file[0] in le1_error_dic:
			le1_error_dic[file[0]] = [0,0]
		record_list = read_file_lines(records_path+file)
		for line in record_list:
			if 'le1 result' in line:
				line_list = line.split(':')
				le1 = line_list[1].strip()
				if le1 != file[0]:
					le1_error_dic[file[0]][1] += 1
				else:
					le1_error_dic[file[0]][0] += 1
	
	print(le1_error_dic)
	test_text_count,error_text_count = 0,0
	rate_of_error_dic = {}
	for i,j in le1_error_dic.items():
		rate_of_error_dic[i] = '{:.3}'.format(j[1]/j[0]*100)+'%'
		test_text_count += j[0]
		error_text_count += j[1]

	save_file(le1_save_path+'le1_error_rate.txt','一级分类错误文档数统计\n\n','w')
	save_file(le1_save_path+'le1_error_rate.txt',str(le1_error_dic)+'\n'+str(rate_of_error_dic)+'\n\n','a')
	le1_error_rate = error_text_count/test_text_count
	save_file(le1_save_path+'le1_error_rate.txt','测试集总数：{}'.format(test_text_count)+'\n'+'错误总数：{}'.format(error_text_count)+'\n'+'错误占比：{:.4}'.format(le1_error_rate),'a')

def main():
	le1_path = '../data/info/category/level_1.txt'
	result_path = '../data/result/level_3_big/final_test/svm_test_way_le2_no-KB.txt'
	save_path = '../data/result/level_3_big/final_test/stat_le3_acc_of_way_le2_no_KB/'
	threshold_list = [0.5,0.4,0.3,0.2,0.1,0.0]
	for th in threshold_list:
		stat_le3_acc(le1_path,result_path,save_path,th)

	records_path = '../data/result/final_test/records_way_8_no-fast/'
	le1_save_path = '../data/result/final_test/stat_le1_acc_of_way_8_no_fast/'
	# stat_le1_error(records_path,le1_save_path)

if __name__ == '__main__':
	main()
