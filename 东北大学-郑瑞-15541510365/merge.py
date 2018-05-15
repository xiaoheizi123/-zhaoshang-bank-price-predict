# -*- coding: utf-8 -*-
import numpy as np


#将不同结果进行融合，以期求得更好的效果，，，的脚本实验文件
with open('9678376_1525922238892_submit.txt','r') as f1:
	data1=f1.readlines()
	list_date=data1
f1.close()
print(type(data1[1][1]))
print(type(np.float64(data1[1][1])))
with open('Y11.txt','r') as f2:
#with open('sublime_Y3_lin.txt','r') as f2:
	data2=f2.readlines()
	list_date=data2
f2.close()
with open("merge_two.txt","w") as f:
	i=0
	for i in range(len(data1)):
		if((data1[i].strip()=="Y1") |(data1[i].strip() =="Y2") | (data1[i].strip() =="Y3") | (data1[i].strip() =="iron")):
			f.write(data1[i])
			print(data1[i])
			i=i+1
		else:
			if(i<=1115 | i>=1667):
				f.write((data1[i].strip().split("\t"))[0]+"\t"+str((data1[i].strip().split("\t"))[1])+"\n")
				i=i+1
			else:
				print(i)
				p=(data1[i].strip().split("\t"))[1]
				print(p)
				print(float(p))
				q=(data2[i].strip().split("\t"))[1]
				
				#temp=float(q)-0.5

				temp=(float(p)*0.2+float(q)*0.8)
				#temp_str="%f"%temp
				f.write((data1[i].strip().split("\t"))[0]+"\t"+str(temp)+"\n")
				i=i+1
f.close()

			
