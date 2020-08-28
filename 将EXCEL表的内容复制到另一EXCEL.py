

#**********************************************************************
import os 
import xlwt,xlrd
from xlutils.copy import copy

last_day_stock_names = os.listdir('./last_day/')

data_10_names = os.listdir('./data_10/')

for last_day_stock_name in last_day_stock_names:
    for data_10_name in data_10_names:
        if data_10_name == last_day_stock_name :
          #获取原EXCEL信息
          wb_temp=xlrd.open_workbook('./last_day/' + last_day_stock_name) # 打开待复制的表
          sheets = wb_temp.sheet_names() # 获取工作簿中的所有工作表名字，形成列表元素
          print("**:",sheets)
          # print(sheets)#打印查看一下aa文件中存在的sheet页名字列表
          sheet1 = wb_temp.sheet_by_index(0) # 根据索引获取第一个sheet
          #打印查看获得的sheet，所有工作表中的的第一个工作表，是一个对象
          # print(sheet1)
          row=sheet1.row_values(1)#获取第二行的内容，row为一个list
          #打印查看一下获得的第二行数据
          # print(row)
          #worksheet= workbook.sheet_by_name(sheets[0]) # 通过sheets[0]工作表名称获取工作簿中所有工作表中的的第一个工作表
          k=sheet1.nrows # 获取第一个工作表中已存在的数据的行数

          #处理要写入的excel工作簿的信息：
          workbook = xlrd.open_workbook('./data_10/'+data_10_name) # 打开工作簿
          new_workbook = copy(workbook) # 将xlrd对象拷贝转化为xlwt对象
          new_worksheet = new_workbook.get_sheet(0) # 获取转化后工作簿中的第一个工作表对象
          #打印查看一下
          # print(new_worksheet,new_workbook,new_worksheet.name)

          # 写入数据到excle中
          for i,content in enumerate(row):
              new_worksheet.write(k-1,i,content)#在获得的第一个sheet对象中，第k行，第i列写入content


          # for i,content in enumerate(row):
          #     new_worksheet.write(3,i,content)#在获得的第一个sheet对象中，第4行，第i列写入content

          new_workbook.save('./data_10/'+data_10_name) # 保存工作簿,都是保存到当前运行目录，保存到指定目录中加上路径就好。




#*************delete excel row or column************************
# coding=utf-8
# #引入openpyxl模块
# import openpyxl
# # 让wb=工作薄，已有的工作薄路径
# wb = openpyxl.load_workbook("./test/e/000001.SZ.xlsx")
# # 让ws等于工作薄中的Sheet1工作表
# ws = wb['sheet']

# # 删除从第3行开始算的2行内容
# ws.delete_rows(2,1)
# # 删除从第1列开始算的2列内容
# # ws.delete_cols(1,2)
# # 另存为表格
# wb.save("./e/000001.SZ.xlsx")


#*************************************************************
























