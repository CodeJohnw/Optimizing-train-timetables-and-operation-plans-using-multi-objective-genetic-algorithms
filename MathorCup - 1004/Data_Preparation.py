import numpy as np
import pandas as pd



######################################################读取文件
'''读取附件1车站数据，获取可作为交路起止点车站列表'''
# 读取Excel文件
file_path_station = '附件1：车站数据.xlsx'
df_station = pd.read_excel(file_path_station)
# 筛选“是”作为交路起终点的车站列表
start_end_stations = [index + 1 for index in df_station[df_station['是否可作为交路起点/终点'] == '是'].index.tolist()]
# 筛选“是”作为交路起终点的车站列表

valid_combinations = []
# print(start_end_stations)
for sa in start_end_stations:
    for sb in start_end_stations:
        if sb - sa >= 3 and sb - sa <= 24:
            valid_combinations.append((sa, sb))


'''读取区间运行时间和距离数据'''
file_path_block = '附件2：区间运行时间.xlsx'
df_block = pd.read_excel(file_path_block)
# 获取区间运行时间和距离数据
next_station_time = df_block['区间运行时间/s'].tolist()
next_station_distance = df_block['站间距/km'].tolist()

'''读取附件3 OD客流数据，并计算六类乘客数量以及乘客总的等待时间'''
file_path_OD = '附件3：OD客流数据.xlsx'
df_OD = pd.read_excel(file_path_OD, index_col=0).fillna(0)

'''读取附件4断面客流数据'''
# 读取Excel文件
file_path4 = '附件4：断面客流数据.xlsx'
df4 = pd.read_excel(file_path4)

# 将指定列数据转换为列表
section_flow_data = df4.iloc[:, 1].tolist()

# 创建一个新的数据框用于存储每个车站的上车和下车总人数
station_totals = pd.DataFrame(index=df_OD.index, columns=["上车总人数", "下车总人数"])

# 计算每个车站上车和下车总人数
for station in df_OD.index:
    # 计算每个车站的上车总人数（出站）
    station_board_passenger = station_totals.at[station, "上车总人数"] = df_OD.loc[station].sum()

    # 计算每个车站的下车总人数（入站）
    station_alight_passenger = station_totals.at[station, "下车总人数"] = df_OD[station].sum()
sta_passenger_board = station_totals[ "上车总人数"].tolist()
sta_passenger_alight = station_totals[ "下车总人数"].tolist()









# #    Bigloop_Train_num, smallloop_Train_num, Sa, Sb, headway, dwell_time, Llb, Lla, Slb, Sla = x
# ga = GA(func=objective_function, n_dim=155, size_pop=50, max_iter=800, 
#         lb=[1, 1, 1, 1, 120]+[20]*30+[1]*30+[1]*30+[1]*30+[1]*30, ub=[30, 30, 30, 30, 360]+[120]*30+[1860]*30+[1860]*30+[1860]*30+[1860]*30,
#         constraint_eq=constraint_eq ,constraint_ueq = constraint_ueq,
#         precision=[1]*155,prob_mut = 0.001)
# best_x, best_y = ga.run()
# print('best_x:', best_x, '\n', 'best_y:', best_y)