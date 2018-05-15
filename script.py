# 	money_get=[]
# 	print(offer_data.cell_value(3,7))
# 	for i in range(3,171):
# 		money_get_larger_conpany=[]
# 		for j in range(6,8):                   #get（6，7）
# 			if(j==6):
# 				ctype = offer_data.cell(i, 0).ctype
# 				cell=offer_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				money_get_larger_conpany.append(cell)
# 			else:
# 				money_get_larger_conpany.append(offer_data.cell_value(i,j))
# 		money_get.append(money_get_larger_conpany)

# 	electrity_fire_percent=[]
# 	for i in range(3,35):
# 		electrity_fire_percent_sub=[]
# 		for j in range(9,12):
# 			if(j==9):
# 				ctype = offer_data.cell(i, 0).ctype
# 				cell=offer_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				money_get_larger_conpany.append(cell)
# 			else:
# 				money_get_larger_conpany.append(offer_data.cell_value(i,j))
# 		electrity_fire_percent.append(electrity_fire_percent_sub)

# 	rough_iron_amount_percent=[]
# 	for i in range(3,315):
# 		rough_iron_amount_percent_sub=[]
# 		for j in range(13,16):
# 			if(j==13):
# 				ctype = offer_data.cell(i, 0).ctype
# 				cell=offer_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				rough_iron_amount_percent_sub.append(cell)
# 			else:
# 				rough_iron_amount_percent_sub.append(offer_data.cell_value(i,j))
# 		rough_iron_amount_percent.append(rough_iron_amount_percent_sub)



# 	#需求数据的处理，此处放弃“我国建筑用钢材消费量”和“国内钢筋消费量”——————这里可以再组织一波变量，求一波结果 \

# 	total_sizes=[]
# 	for i in range(3,159):
# 		total_sizes_sub=[]
# 		for j in range(4):
# 			if(j==0):
# 				ctype = need_data.cell(i, 0).ctype
# 				cell=need_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				total_sizes_sub.append(cell)
# 			else:
# 				total_sizes_sub.append(offer_data.cell_value(i,j))
# 		total_sizes.append(total_sizes_sub)


# 	producity_of_inchina_percent=[]
# 	for i in range(3,171):
# 		producity_of_inchina_percent_sub=[]
# 		for j in range(14,16):
# 				if(j==14):
# 					ctype = need_data.cell(i, 0).ctype
# 					cell=need_data.cell_value(i,0)
# 					if ctype == 2 and cell % 1 == 0:
# 						cell = int(cell)
# 					elif ctype == 3:
# 						date_value =xlrd.xldate_as_tuple(cell,0)
# 						cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 					elif ctype == 4:
# 						cell = True if cell == 1 else False
# 					#print(cell)
# 					producity_of_inchina_percent_sub.append(cell)
# 				else:
# 					producity_of_inchina_percent_sub.append(offer_data.cell_value(i,j))
# 		producity_of_inchina_percent.append(producity_of_inchina_percent_sub)


# 	producty_of_inchina_sametime_percent=[]
# 	for i in range(3,159):
# 		producty_of_inchina_sametime_percent_sub=[]
# 		for j in range(17,19):
# 			if(j==17):
# 				ctype = need_data.cell(i, 0).ctype
# 				cell=need_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				producty_of_inchina_sametime_percent_sub.append(cell)
# 			else:
# 				producty_of_inchina_sametime_percent_sub.append(offer_data.cell_value(i,j))
# 		producty_of_inchina_sametime_percent.append(producty_of_inchina_sametime_percent_sub)


# 	in_out_price_differece=[]
# 	for i in range(3,189):
# 		in_out_price_differece_sub=[]
# 		for j in range(20,22):
# 			if(j==20):
# 				ctype = need_data.cell(i, 0).ctype
# 				cell=need_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				in_out_price_differece_sub.append(cell)
# 			else:
# 				in_out_price_differece_sub.append(offer_data.cell_value(i,j))
# 		in_out_price_differece.append(in_out_price_differece_sub)

# 	in_out_amount_price=[]
# 	for i in range(3,275):
# 		in_out_amount_price_sub=[]
# 		for j in range(24,26):
# 			if(j==24):
# 				ctype = need_data.cell(i, 0).ctype
# 				cell=need_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				in_out_amount_price_sub.append(cell)
# 			else:
# 				in_out_amount_price_sub.append(offer_data.cell_value(i,j))
# 		in_out_amount_price.append(in_out_amount_price_sub)


# 	#成本数据
# 	stone_producity=[]
# 	for i in range(3,363):
# 		stone_producity_sub=[]
# 		for j in range(2):
# 			if(j==0):
# 				ctype = cost_data.cell(i, 0).ctype
# 				cell=cost_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				stone_producity_sub.append(cell)
# 			else:
# 				stone_producity_sub.append(offer_data.cell_value(i,j))
# 		stone_producity.append(stone_producity_sub)

# 	import_stone_amount=[]
# 	for i in range(16,279):
# 		import_stone_amount_sub=[]
# 		for j in range(2,5):
# 			if(j==2):
# 				ctype = cost_data.cell(i, 0).ctype
# 				cell=cost_data.cell_value(i,0)
# 				if ctype == 2 and cell % 1 == 0:
# 					cell = int(cell)
# 				elif ctype == 3:
# 					date_value =xlrd.xldate_as_tuple(cell,0)
# 					cell = date(*date_value[:3]).strftime('%Y/%m/%d')
# 				elif ctype == 4:
# 					cell = True if cell == 1 else False
# 				#print(cell)
# 				import_stone_amount_sub.append(cell)
# 			else:
# 				import_stone_amount_sub.append(offer_data.cell_value(i,j))
# 		import_stone_amount.append(import_stone_amount_sub)


# #至此，训练集的已经统计完毕