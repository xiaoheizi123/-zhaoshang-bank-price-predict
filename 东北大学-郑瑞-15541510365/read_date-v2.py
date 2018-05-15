# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn import  preprocessing
import xlrd
from datetime import date,datetime
import xgboost as xgb
import lightgbm as lgb
#https://github.com/WordZzzz/tianchi_location/blob/master/process/wifi_simple_model_64.py   baseline
#https://zhuanlan.zhihu.com/p/31182879     启发编程
#https://blog.csdn.net/vitodi/article/details/60141301  最好的
#https://datascience.stackexchange.com/questions/17282/xgbregressor-vs-xgboost-train-huge-speed-difference  参数设置
#https://blog.csdn.net/wzmsltw/article/details/50994481   调参指南
#http://xiaoyb.me/2017/06/12/Xgboost%E5%8F%82%E6%95%B0/   推荐的官网
path = "./"
def read_data_train():

	X_train = xlrd.open_workbook("E:/机器学习比赛/X_train.xls")
	
	print("^^^^^^^^^^^")
	#print(X_train.sheet_names())
	sheet_train=X_train.sheet_by_index(0)
	
	print(sheet_train.name)
	#print(sheet_train.nrows)
	#print (sheet_train.ncols)
	#print(sheet_train.rows(0))
	rows = sheet_train.row_values(0)
	date_content=[]
	
	xtrain_list=[]
	
	#print(sheet_train.row_values(0))
	
	for i in range(3018,7766):
		item_content=[]
		for j in range(45):
			
			if(j==0):
				ctype = sheet_train.cell(i, 0).ctype
				#cell = sheet_train.cell(i,0).value
				cell=sheet_train.cell_value(i,0)
				if ctype == 2 and cell % 1 == 0:
					cell = int(cell)
				elif ctype == 3:
					date_value =xlrd.xldate_as_tuple(cell,0)
					#date=datetime(*xlrd.xldate_as_tuple(cell,0))
					cell = date(*date_value[:3]).strftime('%Y-%m-%d')
				elif ctype == 4:
					cell = True if cell == 1 else False
				#print(cell)
				item_content.append(cell)
			elif(sheet_train.cell_value(0,j) in ['GDP:实际同比增长:全球','美国:GDP:现价:环比折年率:季调','欧元区:GDP平减指数:季调',\
				'工业增加值:当月同比','工业增加值:累计同比','固定资产投资完成额:累计同比',\
				'房地产开发投资完成额:累计同比','产量:发电量:当月同比','产量:发电量:累计同比','M0:同比','M1:同比','M2:同比','中债国债到期收益率:3个月',\
				'中债国债到期收益率:1个月','中债国债到期收益率:1年','中债国债到期收益率:5年','中债国债到期收益率:10年','CPI:当月同比',\
				'PPI:全部工业品:当月同比','RPI:当月同比','RPI:累计同比','上证综合指数','沪深300指数',\
				 '中债总指数','中债国债总指数','中债金融债券总指数']):

				#if sheet_train.cell_value(i,j) is None:
				if sheet_train.cell_value(i,j) == '':
					for m in range(366):
						if (i+m <7766):
							#print(i+m)
							if sheet_train.cell_value(i+m,j) !='':
								item_content.append(sheet_train.cell_value(i+m,j) )
								break
				else:
					item_content.append(sheet_train.cell_value(i,j))
		xtrain_list.append(item_content)
	return xtrain_list

def read_data_test():
	X_test = xlrd.open_workbook("E:/机器学习比赛/X_test.xls")
	sheet_test=X_test.sheet_by_index(0)
	
	
	xtest_list=[]
	
	#print(sheet_test.row_values(0))
	for i in range(2,826):
		item_content_test=[]
		y_num=0
		for j in range(45):
			y_num=y_num+1
			if(j==0):
				ctype = sheet_test.cell(i, 0).ctype
				#cell = sheet_train.cell(i,0).value
				cell=sheet_test.cell_value(i,0)
				if ctype == 2 and cell % 1 == 0:
					cell = int(cell)
				elif ctype == 3:
					date_value =xlrd.xldate_as_tuple(cell,0)
					#date=datetime(*xlrd.xldate_as_tuple(cell,0))
					cell = date(*date_value[:3]).strftime('%Y-%m-%d')
				elif ctype == 4:
					cell = True if cell == 1 else False
				item_content_test.append(cell)
			elif(sheet_test.cell_value(0,j) in ['GDP:实际同比增长:全球','美国:GDP:现价:环比折年率:季调','欧元区:GDP平减指数:季调',\
				'工业增加值:当月同比','工业增加值:累计同比','固定资产投资完成额:累计同比',\
				'房地产开发投资完成额:累计同比','产量:发电量:当月同比','产量:发电量:累计同比','M0:同比','M1:同比','M2:同比','中债国债到期收益率:3个月',\
				'中债国债到期收益率:1个月','中债国债到期收益率:1年','中债国债到期收益率:5年','中债国债到期收益率:10年','CPI:当月同比',\
				'PPI:全部工业品:当月同比','RPI:当月同比','RPI:累计同比','上证综合指数','沪深300指数',\
				 '中债总指数','中债国债总指数','中债金融债券总指数']):
				temp=-99
				if sheet_test.cell_value(i,j)=='':
					M=0
					p=0
					for m in range(366):
						if (i+m<826):
							if sheet_test.cell_value(i+m,j) != '':
								item_content_test.append(sheet_test.cell_value(i+m,j) )
								temp=sheet_test.cell_value(i+m,j)
								M=1
								break

					# if(temp==-99):
					# 	if(len(item_content_test)<y_num):
					# 		item_content_test.append(xtest_list[i-1][j])
					# 	else:
					# 		item_content_test[y_num-1]=xtest_list[i-1][j]
					# 	M=1
					# 	break
				else:
					item_content_test.append(sheet_test.cell_value(i,j) )
		xtest_list.append(item_content_test)
	return xtest_list


def train_predict():
	Y_train = xlrd.open_workbook("E:/机器学习比赛/Y_train.xls")
	train_predict1_Y = Y_train.sheet_by_index(0)
	predict_Y=[]
	for i in range(3392):
		predict1_Y=[]
		for j in range(2):			
			if(j==0):
				ctype = train_predict1_Y.cell(i, 0).ctype
				#cell = sheet_train.cell(i,0).value
				cell=train_predict1_Y.cell_value(i,0)
				if ctype == 2 and cell % 1 == 0:
					cell = int(cell)
				elif ctype == 3:
					date_value =xlrd.xldate_as_tuple(cell,0)
					#date=datetime(*xlrd.xldate_as_tuple(cell,0))
					cell = date(*date_value[:3]).strftime('%Y-%m-%d')
				elif ctype == 4:
					cell = True if cell == 1 else False
				predict1_Y.append(cell)
		else:
			predict1_Y.append(train_predict1_Y.cell_value(i,j))
		predict_Y.append(predict1_Y)
	return predict_Y

def train_predictY2():
	Y_train = xlrd.open_workbook("E:/机器学习比赛/Y_train.xls")
	train_predict_Y2 = Y_train.sheet_by_index(1)
	print(type(train_predict_Y2))
	predict_Y2=[]
	for i in range(3494):
		predict2_Y=[]
		for j in range(2):
			if(j==0):
				ctype = train_predict_Y2.cell(i, 0).ctype
				cell=train_predict_Y2.cell_value(i,0)
				if ctype == 2 and cell % 1 == 0:
					cell = int(cell)
				elif ctype == 3:
					date_value =xlrd.xldate_as_tuple(cell,0)
					#date=datetime(*xlrd.xldate_as_tuple(cell,0))
					cell = date(*date_value[:3]).strftime('%Y-%m-%d')
				elif ctype == 4:
					cell = True if cell == 1 else False
				predict2_Y.append(cell)
		else:
			predict2_Y.append(train_predict_Y2.cell_value(i,j))
		predict_Y2.append(predict2_Y)
	return predict_Y2

def train_predictY3():
	Y_train = xlrd.open_workbook("E:/机器学习比赛/Y_train.xls")
	train_predict_Y3 = Y_train.sheet_by_index(2)
	print(type(train_predict_Y3))
	predict_Y3=[]
	for i in range(2822):
		predict3_Y=[]
		for j in range(2):
			if(j==0):
				ctype = train_predict_Y3.cell(i, 0).ctype
				cell=train_predict_Y3.cell_value(i,0)
				if ctype == 2 and cell % 1 == 0:
					cell = int(cell)
				elif ctype == 3:
					date_value =xlrd.xldate_as_tuple(cell,0)
					#date=datetime(*xlrd.xldate_as_tuple(cell,0))
					cell = date(*date_value[:3]).strftime('%Y-%m-%d')
				elif ctype == 4:
					cell = True if cell == 1 else False
				predict3_Y.append(cell)
		else:
			predict3_Y.append(train_predict_Y3.cell_value(i,j))
		predict_Y3.append(predict3_Y)
	return predict_Y3



if __name__=="__main__":
	data_list=read_data_train()
	#print(data_list)
	data_list_test=read_data_test()
	print(len(data_list_test))
	data_predict_Y=train_predict()
	print(len(data_predict_Y))
	#print(data_predict_Y)
	data_predict_Y2=train_predictY2()
	print(len(data_predict_Y2))

	data_predict_Y3=train_predictY3()
	print(len(data_predict_Y3))


	#将训练数据从list转变为numpy.array(),方便进行转化成数据格式DMatrix，
	a_train_Y=len(data_predict_Y)
	a_train_Y2=len(data_predict_Y2)
	a_train_Y3=len(data_predict_Y3)
	a_train_X=len(data_list)
	b_train_X=len(data_list[0])-1
	print(a_train_Y3)
	print(b_train_X)
	valid_data_for_Ytrain=np.zeros((a_train_Y3,b_train_X))            #这里每当换数据集时，训练集需要响应的改变，for Y2,(a_train_Y2,b_train_X);Y3 (a_train_Y3,b_train_X)
	list_valid_data_for_Ytrain=[]
	ii=0
	jj=0
	for ii in range(a_train_Y3):                                      #
		temp_date=data_predict_Y3[ii][0]                              #这里每当还数据集时，训练集需要响应的改变
		#print(data_predict_Y3[ii][0])                                  #
		for jj in range(a_train_X):
			if(data_list[jj][0]==temp_date):                #改
				#print(jj)
				# print(data_predict_Y2[ii][0])
				#print(data_list[jj][:])
				#print(temp_date)
				#print("^^^^^^^^^^^^^")
				valid_data_for_Ytrain[ii]=data_list[jj][1:]
				# print(data_list[jj])
				# print("3434343")
				list_valid_data_for_Ytrain.append(data_list[jj][1:])
				jj=jj+1
				break
			else:
				jj=jj+1
		ii=ii+1
	#print(list_valid_data_for_Ytrain)


	#将Y1的预测数据转化为numpy格式，便于训练
	valid_predict_for_Ytrain=np.zeros((a_train_Y,1))
	list_valid_predict_for_Ytrain=[]
	for i in range(a_train_Y):
		valid_predict_for_Ytrain[i]= data_predict_Y[i][1]
		list_valid_predict_for_Ytrain.append(data_predict_Y[i][1])
	#print(valid_data_for_Ytrain[1])

	
	#将测试数据进行数据格式转化，可以对次数列数据进行预测，当然此序列数据不同于提交数据的序列
	a_test_x=len(data_list_test)
	valid_data_for_Ytest=np.zeros((a_test_x,b_train_X))
	date_valid_data_for_out=[]
	list_valid_data_for_Ytest=[]
	for i in range(a_test_x):	
	 	#print(i)
	 	#print(len(valid_data_for_Ytest[i]))
	 	#print(data_list_test[i][:])
	 	valid_data_for_Ytest[i]=data_list_test[i][1:]
	 	date_valid_data_for_out.append(data_list_test[i][0])
	 	list_valid_data_for_Ytest.append(data_list_test[i][1:])

	datefile=open("datefile_Y2.txt","w")                       #需要修改得地方
	for date in date_valid_data_for_out:
		datefile.write(date)
		datefile.write("\n")
	datefile.close()

	#将Y2的预测数据值转化为numpy格式，训练Y2序列的模型
	valid_predict_for_Y2train=np.zeros((len(data_predict_Y2),1))
	list_valid_predict_for_Y2train=[]
	for i in range(len(data_predict_Y2)):
		valid_predict_for_Y2train[i]= data_predict_Y2[i][1]
		list_valid_predict_for_Y2train.append(data_predict_Y2[i][1])


	#将Y3的预测数据值转化为numpy格式，训练Y2序列的模型
	valid_predict_for_Y3train=np.zeros((len(data_predict_Y3),1))
	list_valid_predict_for_Y3train=[]
	for i in range(len(data_predict_Y3)):
		valid_predict_for_Y3train[i]= data_predict_Y3[i][1]
		list_valid_predict_for_Y3train.append(data_predict_Y3[i][1])



#https://github.com/dmlc/xgboost/blob/master/demo/multiclass_classification/train.py

	max_depth = 6
	min_child_weight = 7
	subsample = 1
	colsample_bytree = 1
	objective = 'reg:linear'
	num_estimators = 1000
	learning_rate = 0.35

	
#http://xgboost.readthedocs.io/en/latest/python/python_api.html




	#xgbtrain = xgb.DMatrix(feature[0])
	# print("&&&&&&&&&&&&&&")
	# print(type(valid_data_for_Ytrain))
	# print(len(valid_predict_for_Ytrain))
	# print(len(valid_data_for_Ytest))
	# xgbtrain = xgb.DMatrix(valid_data_for_Ytrain)
	# y_values=xgb.DMatrix(valid_predict_for_Ytrain)

	# xgbtest = xgb.DMatrix(valid_data_for_Ytest)

	# bst=xgb.XGBRegressor(max_depth=max_depth,
	# 	min_child_weight=min_child_weight,
	# 	subsample=subsample,
	# 	colsample_bytree=colsample_bytree,
	# 	objective=objective,
	# 	n_estimators=num_estimators,
	# 	learning_rate=learning_rate)
	# clf1=LinearRegression(fit_intercept=True)
	# clf1.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Ytrain)
	# pred_L1=clf1.predict(list_valid_data_for_Ytest)
	# print(type(pred_L1))
	# print(pred_L1[2])
	# print(pred_L1[7])


	# rfr1=RandomForestRegressor()
	# rfr1.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Ytrain)
	# pred_R1=rfr1.predict(list_valid_data_for_Ytest)
	# bst=xgb.XGBRegressor()
	# bst.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Ytrain)
	# preds1=bst.predict(list_valid_data_for_Ytest)
	# print(len(preds1))
	# # counter1=0
	# # for counter1 in range(len(preds1)):
	# # 	preds1[counter1]=(preds1[counter1]+pred_R1[counter1]+pred_L1[counter1])/3
	# # 	counter1=counter1+1
	# result1=pd.DataFrame(preds1)
	# result1.to_csv(path+'sublime.txt',index=False)
	# ceshi_y=bst.predict(list_valid_data_for_Ytrain)
	# ceshi_y_pred=[round(value,5) for value in ceshi_y]
	# print("########")
	# print(ceshi_y_pred[0])
	# print(type(ceshi_y_pred))
	# print("???????????????????????????????????????\n")
	# print(type(list_valid_predict_for_Ytrain))
	# print(list_valid_predict_for_Ytrain[0])
	# popy=[round(value,5) for value in list_valid_predict_for_Ytrain]
	# accuracy=accuracy_score(popy, ceshi_y_pred)
	# print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))






    #选中要注释的代码，按住"Ctr+/" 可以快速注释，再次按下，可以取消注释

    #训练得到Y2的结果
	#Y2_values=xgb.DMatrix(valid_predict_for_Y2train)
	#print(valid_predict_for_Y2train)





	bst2=xgb.XGBRegressor(max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective=objective,
                n_estimators=num_estimators,
                learning_rate=learning_rate)

 # 	clf2=LinearRegression(fit_intercept=True)
	# clf2.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Y2train)
	# pred_L2=clf2.predict(list_valid_data_for_Ytest)
	# print(type(pred_L2))
	# print(pred_L2[2])
	# print(pred_L2[7])


	# rfr2=RandomForestRegressor()
	# rfr2.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Y2train)
	# pred_R2=rfr2.predict(list_valid_data_for_Ytest)

	# print("^^^^^^^^^^^")
	# bst2=xgb.XGBRegressor(seed=1850)
	# print(type(list_valid_data_for_Ytrain))
	# #print(list_valid_data_for_Ytest)
	# bst2.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Y2train)
	# preds2=bst2.predict(list_valid_data_for_Ytest)
	# print(len(preds2))
	# # counter2=0
	# # for counter2 in range(len(preds2)):
	# # 	preds2[counter2]=(preds2[counter2]+pred_R2[counter2]+pred_L2[counter2])/3
	# # 	counter2=counter2+1
	# result2=pd.DataFrame(preds2)
	# result2.to_csv(path+'sublime_Y2.txt',index=False)
	#print(list_valid_data_for_Ytrain)

	#     #训练得到Y3的结果
	# Y3_values=xgb.DMatrix(valid_predict_for_Y3train)
	#print(list_valid_data_for_Ytrain)






	# clf3=LinearRegression(fit_intercept=True)
	# clf3.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Y3train)
	# pred_L3=clf3.predict(list_valid_data_for_Ytest)
	# print(pred_L3)
	# # print(pred_L3[2])
	# # print(pred_L3[7])


	# rfr3=RandomForestRegressor()
	# rfr3.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Y3train)
	# pred_R3=rfr3.predict(list_valid_data_for_Ytest)


	# # bst3=xgb.XGBRegressor(max_depth=max_depth,
	# # 	min_child_weight=min_child_weight,
	# # 	subsample=subsample,
	# # 	colsample_bytree=colsample_bytree,
	# # 	objective=objective,
	# # 	n_estimators=num_estimators,
	# # 	learning_rate=learning_rate)
	bst3=xgb.XGBRegressor(seed=1000)
	#bst3=xgb.XGBRegressor()
	bst3.fit(list_valid_data_for_Ytrain,list_valid_predict_for_Y3train)
	#print(list_valid_data_for_Ytest)
	preds3=bst3.predict(list_valid_data_for_Ytest)
	print(len(preds3))
	#print(preds3[0])
	# counter3=0
	# for counter3 in range(len(preds3)):
	# 	preds3[counter3]=(preds3[counter3]+pred_R3[counter3]+pred_L3[counter3])/3
	# 	counter3=counter3+1

	# print(preds3[0])

	result3=pd.DataFrame(preds3)
	result3.to_csv(path+'sublime_Y3.txt',index=False)
	
	# ceshi_y=clf3.predict(list_valid_data_for_Ytrain)
	# ceshi_y_pred=[round(value) for value in ceshi_y]
	# print("########")
	# print(ceshi_y_pred[0])
	# print(type(ceshi_y_pred))
	# print("???????????????????????????????????????\n")
	# print(type(list_valid_predict_for_Ytrain))
	# print(list_valid_predict_for_Ytrain[0])
	# popy=[round(value) for value in list_valid_predict_for_Y3train]
	# accuracy=accuracy_score(popy, ceshi_y_pred)
	# print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))





	print("xgboost end")


    		



    



 
