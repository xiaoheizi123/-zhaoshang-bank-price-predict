# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import  preprocessing
import xlrd
from datetime import date,datetime
import xgboost as xgb
import lightgbm as lgb


#https://www.zhihu.com/question/26639110  缺失数据处理方法
path = "./"
def read_icon_train():
	icX_train = xlrd.open_workbook("E:/机器学习比赛/iconX_train.xlsx")
	offer_data=icX_train.sheet_by_index(0)
	need_data=icX_train.sheet_by_index(1)
	cost_data=icX_train.sheet_by_index(2)
	
	#试一试集中的表示
	print(offer_data.nrows)
	print(offer_data.ncols)

	#第一列从2008年到2015年，每个值进行复制12个月
	year_first=[]           #国内钢铁行业集中度（前10大钢企粗钢产量合计占比)
	for i in range(7,16):
		temp_value=offer_data.cell_value(i,1)
		for j in range(12):
			year_first.append(temp_value)

	year_second=[]             #'电炉钢占粗钢产量比例'
	for i in range(27,35):
		temp_value1=offer_data.cell_value(i,10)
		temp_value2=offer_data.cell_value(i,11)
		for j in range(12):
			year_second.append(temp_value1)
			year_second.append(temp_value1)


	offer_data_list=[]
	for i in range(3,99):
		offer_data_content=[]
		for j in range(16):	
			if(offer_data.cell_value(0,j) in ['全球粗钢产能利用率', '大中型钢铁企业的吨钢利润总额（元/吨）','国内粗钢产量/生铁产量']):
				if(offer_data.cell_value(0,j) in ['全球粗钢产能利用率']):
					cell=offer_data.cell_value(i,j-1)
					#if(offer_data.cell(i,j-1).ctype==3):
					date_value =xlrd.xldate_as_tuple(cell,0)
					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
					offer_data_content.append(cell)
					offer_data_content.append(offer_data.cell_value(i,j))


				if(offer_data.cell_value(0,j) in ['大中型钢铁企业的吨钢利润总额（元/吨）']):
					offer_data_content.append(offer_data.cell_value(i+72,j))

				if(offer_data.cell_value(0,j) in ['国内粗钢产量/生铁产量']):
					offer_data_content.append(offer_data.cell_value(i+216,j))
					offer_data_content.append(offer_data.cell_value(i+216,j+1))   #'国内焦炭产量/生铁产量'
		offer_data_list.append(offer_data_content)



	year_first_need=[]                                     #'我国建筑用钢材消费量（万吨）'
	for i in range(4,12):
		temp_value1=need_data.cell_value(i,6)
		temp_value2=need_data.cell_value(i,7)
		temp_value3=need_data.cell_value(i,8)
		for j in range(12):
			year_first_need.append(temp_value1)
			year_first_need.append(temp_value2)
			year_first_need.append(temp_value3)


	need_data_list=[]
	for i in range(3,99):
		need_data_content=[]
		for j in range(need_data.ncols):
			if(need_data.cell_value(0,j) in ['我国房地产施工面积（万平方米）','国内钢筋消费量（趋势线）（万吨）',\
				'中国大陆粗钢产量占全球产量比重','中国大陆粗钢产量同比','国内外钢价差（2000/07=0）', \
				'国内钢铁净出口量（百万吨）','国内钢材出口量（万吨）']):
				
				if(need_data.cell_value(0,j) in ['我国房地产施工面积（万平方米）']):
					cell=need_data.cell_value(i+60,j-1)
					#if(offer_data.cell(i,j-1).ctype==3):
					date_value =xlrd.xldate_as_tuple(cell,0)
					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
					need_data_content.append(cell)
					need_data_content.append(need_data.cell_value(i+60,j))
					need_data_content.append(need_data.cell_value(i+60,j+1))
					need_data_content.append(need_data.cell_value(i+60,j+2))


				if(need_data.cell_value(0,j) in ['国内钢筋消费量（趋势线）（万吨）']):
					need_data_content.append(need_data.cell_value(i+21,j))      #'国内线材消费量（趋势线）（万吨）'
					need_data_content.append(need_data.cell_value(i+21,j+1))

				if(need_data.cell_value(0,j) in ['中国大陆粗钢产量占全球产量比重']):
					need_data_content.append(need_data.cell_value(i+72,j))

				if(need_data.cell_value(0,j) in ['中国大陆粗钢产量同比']):
					need_data_content.append(need_data.cell_value(i+60,j))
				if(need_data.cell_value(0,j) in ['国内外钢价差（2000/07=0）']):
					need_data_content.append(need_data.cell_value(i+90,j))
				if(need_data.cell_value(0,j) in ['国内钢铁净出口量（百万吨）']):
					need_data_content.append(need_data.cell_value(i+90,j))

				if(need_data.cell_value(0,j) in ['国内钢材出口量（万吨）']):
					need_data_content.append(need_data.cell_value(i+176,j))
				#print(need_data_content)
		need_data_list.append(need_data_content)

	cost_data_list=[]
	for i in range(3,99):
		cost_data_content=[]
		for j in range(cost_data.ncols):
			if(cost_data.cell_value(0,j) in ['我国铁矿石原矿产量（万吨）','我国进口铁矿石合计（万吨）','我国进口铁矿石同比']):
				if(cost_data.cell_value(0,j) in ['我国铁矿石原矿产量（万吨）']):
					cell=cost_data.cell_value(i+264,j-1)
					#if(offer_data.cell(i,j-1).ctype==3):
					date_value =xlrd.xldate_as_tuple(cell,0)
					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
					cost_data_content.append(cell)
					cost_data_content.append(cost_data.cell_value(i+264,j))


				if(cost_data.cell_value(0,j) in ['我国进口铁矿石合计（万吨）']):
					cost_data_content.append(cost_data.cell_value(i+180,j))
					#print(cost_data_content)

				if(cost_data.cell_value(0,j) in ['我国进口铁矿石同比']):
					cost_data_content.append(cost_data.cell_value(i+180,j))
		cost_data_list.append(cost_data_content)

	print(len(cost_data_list))
	print("!!!!!!!!!!!")
	print(len(need_data_list))
	#print(need_data_list)
	print("***********")
	print(len(offer_data_list))
	#print(offer_data_list)

	#得到最后的训练数据
	final_train=[]
	#print(year_first)
	for i in range(96):
		final_train_sub=[]
		final_train_sub.extend(offer_data_list[i][1:])
		final_train_sub.extend(need_data_list[i][1:])
		final_train_sub.extend(cost_data_list[i][1:])
		final_train_sub.append(year_first[i])
		final_train_sub.append(year_first_need[i])
		#print(final_train_sub)
		#print(i)
		final_train.append(final_train_sub)
	return final_train
	



def read_icon_test():
	icX_test = xlrd.open_workbook("E:/机器学习比赛/icon_test.xlsx")
	offer_data=icX_test.sheet_by_index(0)
	need_data=icX_test.sheet_by_index(1)
	cost_data=icX_test.sheet_by_index(2)

	year_first=[]           #国内钢铁行业集中度（前10大钢企粗钢产量合计占比)
	for i in range(3,5):
		temp_value=offer_data.cell_value(i,1)
		for j in range(12):
			year_first.append(temp_value)
	print(len(year_first))

	year_second=[]             #'电炉钢占粗钢产量比例'
	for i in range(3,4):
		temp_value1=offer_data.cell_value(i,10)
		temp_value2=offer_data.cell_value(i,11)
		for j in range(12):
			year_second.append(temp_value1)
			year_second.append(temp_value1)


	offer_data_list=[]
	for i in range(3,27):
		offer_data_content=[]
		for j in range(16):	
			if(offer_data.cell_value(0,j) in ['全球粗钢产能利用率', '大中型钢铁企业的吨钢利润总额（元/吨）','国内粗钢产量/生铁产量']):
				if(offer_data.cell_value(0,j) in ['全球粗钢产能利用率']):
					cell=offer_data.cell_value(i,j-1)
					#if(offer_data.cell(i,j-1).ctype==3):
					date_value =xlrd.xldate_as_tuple(cell,0)
					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
					offer_data_content.append(cell)
					offer_data_content.append(offer_data.cell_value(i,j))

				if(offer_data.cell_value(0,j) in ['大中型钢铁企业的吨钢利润总额（元/吨）']):
					if(offer_data.cell_value(i,j)==''):
						if(offer_data.cell_value(i-1,j) !=''):
							offer_data_content.append(offer_data.cell_value(i-1,j))
						else:
							offer_data_content.append(offer_data.cell_value(i-2,j))

					else:
						offer_data_content.append(offer_data.cell_value(i,j))

				if(offer_data.cell_value(0,j) in ['国内粗钢产量/生铁产量']):
					offer_data_content.append(offer_data.cell_value(i,j))
					offer_data_content.append(offer_data.cell_value(i,j+1))   #'国内焦炭产量/生铁产量'
		offer_data_list.append(offer_data_content)



	year_first_need=[]                                     #'我国建筑用钢材消费量（万吨）'
	for i in range(3,5):
		temp_value1=need_data.cell_value(i,6)
		temp_value2=need_data.cell_value(i,7)
		temp_value3=need_data.cell_value(i,8)
		for j in range(12):
			year_first_need.append(temp_value1)
			year_first_need.append(temp_value2)
			year_first_need.append(temp_value3)


	need_data_list=[]
	for i in range(3,27):
		need_data_content=[]
		for j in range(need_data.ncols):
			if(need_data.cell_value(0,j) in ['我国房地产施工面积（万平方米）','国内钢筋消费量（趋势线）（万吨）',\
				'中国大陆粗钢产量占全球产量比重','中国大陆粗钢产量同比','国内外钢价差（2000/07=0）', \
				'国内钢铁净出口量（百万吨）','国内钢材出口量（万吨）']):
				
				if(need_data.cell_value(0,j) in ['我国房地产施工面积（万平方米）']):
					cell=need_data.cell_value(i,j-1)
					#if(offer_data.cell(i,j-1).ctype==3):
					date_value =xlrd.xldate_as_tuple(cell,0)
					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
					need_data_content.append(cell)
					need_data_content.append(need_data.cell_value(i,j))
					need_data_content.append(need_data.cell_value(i,j+1))
					need_data_content.append(need_data.cell_value(i,j+2))


				if(need_data.cell_value(0,j) in ['国内钢筋消费量（趋势线）（万吨）']):
					need_data_content.append(need_data.cell_value(i,j))      #'国内线材消费量（趋势线）（万吨）'
					need_data_content.append(need_data.cell_value(i,j+1))

				if(need_data.cell_value(0,j) in ['中国大陆粗钢产量占全球产量比重']):
					need_data_content.append(need_data.cell_value(i,j))

				if(need_data.cell_value(0,j) in ['中国大陆粗钢产量同比']):
					need_data_content.append(need_data.cell_value(i,j))
				if(need_data.cell_value(0,j) in ['国内外钢价差（2000/07=0）']):
					need_data_content.append(need_data.cell_value(i,j))
				if(need_data.cell_value(0,j) in ['国内钢铁净出口量（百万吨）']):
					need_data_content.append(need_data.cell_value(i,j))

				if(need_data.cell_value(0,j) in ['国内钢材出口量（万吨）']):
					need_data_content.append(need_data.cell_value(i,j))
				#print(need_data_content)
		need_data_list.append(need_data_content)

	cost_data_list=[]
	for i in range(3,27):
		cost_data_content=[]
		for j in range(cost_data.ncols):
			if(cost_data.cell_value(0,j) in ['我国铁矿石原矿产量（万吨）','我国进口铁矿石合计（万吨）','我国进口铁矿石同比']):
				if(cost_data.cell_value(0,j) in ['我国铁矿石原矿产量（万吨）']):
					cell=cost_data.cell_value(i+360,j-1)
					#if(offer_data.cell(i,j-1).ctype==3):
					date_value =xlrd.xldate_as_tuple(cell,0)
					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
					cost_data_content.append(cell)
					cost_data_content.append(cost_data.cell_value(i+360,j))


				if(cost_data.cell_value(0,j) in ['我国进口铁矿石合计（万吨）']):
					cost_data_content.append(cost_data.cell_value(i+276,j))
					#print(cost_data_content)

				if(cost_data.cell_value(0,j) in ['我国进口铁矿石同比']):
					cost_data_content.append(cost_data.cell_value(i+276,j))
		cost_data_list.append(cost_data_content)

	print(len(cost_data_list))
	print("!!!!!!!!!!!")
	print(len(need_data_list))
	#print(need_data_list)
	print("***********")
	print(len(offer_data_list))
	#print(offer_data_list)

	#得到最后的训练数据
	final_test=[]
	#print(year_first)
	for i in range(24):
		final_test_sub=[]
		final_test_sub.extend(offer_data_list[i][1:])
		final_test_sub.extend(need_data_list[i][1:])
		final_test_sub.extend(cost_data_list[i][1:])
		final_test_sub.append(year_first[i])
		final_test_sub.append(year_first_need[i])
		#print(final_train_sub)
		#print(i)
		final_test.append(final_test_sub)

	return final_test


	#鉴于“全球粗钢产能利用率”年限较短，所以不予考虑

def icon_y():
	y_train = xlrd.open_workbook("E:/机器学习比赛/Y_train.xls")
	train_predict_icon = y_train.sheet_by_index(3)
	#print(type(train_predict_icon))
	predict_icon=[]
	cal_plus=0
	plus_num=0
	predict_icon_sub=[]
	m=0
	for i in range(396,807):			
		cell=train_predict_icon.cell_value(i,0)
		date_value =xlrd.xldate_as_tuple(cell,0)
		if(m==0):
			m=date_value[1]
			cal_plus=cal_plus+train_predict_icon.cell_value(i,1)
			plus_num=plus_num+1
			# print("bagin")
			# print(train_predict_icon.cell_value(i,1))
		else:
			n=date_value[1]
			if(n==m):
				cal_plus=cal_plus+train_predict_icon.cell_value(i,1)
				#print(train_predict_icon.cell_value(i,1))
				plus_num=plus_num+1
				if(i+1<807):
					cell1=train_predict_icon.cell_value(i+1,0)
					valid_date_next=xlrd.xldate_as_tuple(cell1,0)
				if (valid_date_next[1] != n ):
					
					m=0
					predict_icon.append(cal_plus/plus_num)
					#predict_icon.append(predict_icon_sub)
					# print(predict_icon_sub)
					# print(plus_num)
					# print("end!!!!!!!!!")
					newer=date_value[1]
					cal_plus=0
					plus_num=0
					#predict_icon_sub=[]
	predict_icon.append(2.238572)
	return predict_icon

if __name__ =="__main__":
	train_data=read_icon_train()
	test_data=read_icon_test()
	train_y=icon_y()
	#print(train_y)
	#print(len(train_y))

	# a_train_icon=len(train_data)
	# b_train_icon=len(train_data[1])
	# print(b_train_icon)
	# a_test_data=len(test_data)
	# a_train_y=len(train_y)

	# train_data_numpy=np.zeros((a_train_icon,b_train_icon))
	# for i in range(a_train_icon):
	# 	#print(train_data[i])
	# 	train_data_numpy[i]=train_data[i]
	# 	#print(i)
	# #print(train_data_numpy)

	# test_data_numpy=np.zeros((a_test_data,b_train_icon))
	# for i in range(a_test_data):
	# 	test_data_numpy[i]=test_data[i]
	# #print(test_data_numpy)

	# train_y_numpy=np.zeros((a_train_icon,1))
	# for i in range(a_train_icon):
	# 	train_y_numpy[i]=train_y[i]
	# #print(train_y_numpy)


	max_depth = 8
	min_child_weight = 7
	subsample = 1
	colsample_bytree = 1
	objective = 'reg:linear'
	num_estimators = 1000
	learning_rate = 0.2

	bst=xgb.XGBRegressor(max_depth=max_depth,
		min_child_weight=min_child_weight,
		subsample=subsample,
		colsample_bytree=colsample_bytree,
		objective=objective,
		n_estimators=num_estimators,
		learning_rate=learning_rate)
	#bst=xgb.XGBRegressor()
	bst.fit(train_data,train_y)
	preds=bst.predict(test_data)
	print(len(preds))
	result=pd.DataFrame(preds)
	result.to_csv(path+'sublime_icon.txt',index=False)