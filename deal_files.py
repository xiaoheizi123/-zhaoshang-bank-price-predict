# -*- coding: utf-8 -*-
import codecs
#https://segmentfault.com/q/1010000002493464/a-1020000002493529


#将之前程序得到的结果整理的最后的提交文件上去
list_date=[]
list_price=[]
list_to_out=[]
with open('datefile.txt','r') as f1:
	data1=f1.readlines()
	list_date=data1
f1.close()

with open("sublime.txt","r") as f2:
	data2=f2.readlines()
	list_price=data2
f2.close()

with open("submit_sample.txt","r") as f3:
	data3=f3.readlines()
	list_sample=data3
f3.close()

with open("sublime_Y2.txt","r") as f6:
	data6=f6.readlines()
	list_sample=data6
f6.close()
print("Y2长度")
#print(data6)

with open("sublime_Y3.txt","r") as f7:   #gai
	data7=f7.readlines()
	list_sample=data7
f7.close()
print("Y3长度")
print(len(data7))

with open("datefile_Y2.txt","r") as f8:
	data8=f8.readlines()
	list_sample=data8
f8.close()
print("Y2日期长度")
print(len(data8))

with open("datefile_Y3.txt","r") as f9:
	data9=f9.readlines()
	list_sample=data9
f9.close()

with open("sublime_icon.txt","r") as f10:    #个数与提交数据数量不对，注意扩张
	data10=f10.readlines()
	list_sample=data10
f10.close()
#print(data10)

with open("Y11.txt","r") as f11:
	data11=f11.readlines()  
	list_sample=data11
f11.close()


print(len(data3))
print(len(data1))
counter1=0
# print(data3[0])
# print(type(data3[0]))
# print(str(data3[0])=='Y1\n')
# print(type('Y1'))
for i in range(len(data3)):
	counter1=counter1+1
	needed_date=data3[i]
	if(needed_date == "Y2\n"):
		print(i)
		print("i")
		break                               #打破了最小封闭for或while循环
counter1=counter1-2
print("counter1")
print(counter1)

counter2=0
for i in range(len(data3)):
	counter2=counter2+1
	needed_date=data3[i]
	if(needed_date == "Y3\n"):
		break
print("counter2")
#print(len(data3))
counter2=counter2-3-counter1
print(counter2)


counter3=0
for i in range(len(data3)):
	needed_date=data3[i]
	counter3=counter3+1
	if(needed_date == "iron\n"):
		break
counter3=counter3-4-counter2-counter1
print("counter3")
print(counter3)


#f4=codecs.open("Y.txt","w")
with open("Y.txt","w") as f4:
	f4.write('Y1\n')
	i=0
	j=0
	for i in range(counter1):
		#print(counter1)
		needed_date=data3[i+1]
		for j in range(len(data1)):
			if (data1[j] == needed_date):
				f4.write(data1[j].strip()+'\t'+str(data2[j+1]))
				break
			else:
				j=j+1
		i=i+1

	f4.write("Y2\n")
	print("~~~~~~~~~~~~~~~~~~~~~~~")
	print(i)
	m=0
	n=0
	import random
	print(counter2)
	for m in range(counter2):
		#print(counter2)
		needed_date=data3[counter1+2+m]
		# print(needed_date.strip())
		# print("222222222222222")
		# print(data11[counter1+2+m].strip().split("\t")[0])
		# print((data11[counter1+2+m].strip().split("\t"))[1])
		# t=(data11[counter1+2+m].strip().split("\t"))[1]
		# t=str(float(t)+random.randint(100,10000)/10000000000)
		# if(needed_date.strip()==(data11[counter1+2+m].strip().split("\t"))[0]):
			
		# 	f4.write(data11[counter1+2+m].strip().split("\t")[0]+'\t'+str(t)+"\n")
		# 	print("55555555555555555")
		for n in range(len(data8)):
			if (data8[n] == needed_date):
				f4.write(data8[n].strip()+'\t'+str(data6[n+1]))
				break
			else:
				n=n+1
		m=m+1


	f4.write('Y3\n')
	print("********************************************")
	w=0
	s=0
	for w in range(counter3):
		needed_date=data3[counter1+counter2+3+w]
		print(data11[counter1+counter2+3+w].strip().split("\t")[0])
		print((data11[counter1+counter2+3+w].strip().split("\t"))[1])
		t=(data11[counter1+counter2+3+w].strip().split("\t"))[1]
		t=str(float(t)+random.randint(100,10000)/100000000)
		if(needed_date.strip()==(data11[counter1+counter2+3+w].strip().split("\t"))[0]):
			
			f4.write(data11[counter1+counter2+3+w].strip().split("\t")[0]+'\t'+str(t)+"\n")
			print("55555555555555555")

		
		# for s in range(len(data9)):
		# 	if (data9[s] == needed_date):
		# 		f4.write(data9[s].strip()+'\t'+str(data7[s+1]))
		# 		break
		# 	else:
		# 		s=s+1
		w=w+1

	f4.write('iron\n')
	z=0
	g=0
	b=1
	for g in range(101):
		needed_date=data3[1667+g]
		
		if(z==0):
			f4.write(needed_date.strip()+'\t'+str(data10[b]))
			z=needed_date.strip().split('-')[:2]
		else:
			p=needed_date.strip().split('-')[:2]
			if(p==z):
				f4.write(needed_date.strip()+'\t'+str(data10[b]))
				if(1667+g+1<1768):
					needed_date_next=data3[1667+g+1]
					z_next=needed_date_next.strip().split('-')[:2]
				if(z_next!=z):
					z=0
					b=b+1

		g=g+1



f4.close()
