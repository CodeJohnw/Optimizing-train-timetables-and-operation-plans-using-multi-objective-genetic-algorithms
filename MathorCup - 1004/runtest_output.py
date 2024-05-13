import pandas as pd
import numpy as np
from sko.GA import GA
import matplotlib.pyplot as plt

# 读取所有Excel文件一次，而不是分别读取
file_path_station = '附件1：车站数据.xlsx'
file_path_block = '附件2：区间运行时间.xlsx'
file_path_OD = '附件3：OD客流数据.xlsx'
file_path4 = '附件4：断面客流数据.xlsx'

df_station = pd.read_excel(file_path_station)
df_block = pd.read_excel(file_path_block)
df_OD = pd.read_excel(file_path_OD, index_col=0).fillna(0)
df4 = pd.read_excel(file_path4)

# 将指定列数据转换为列表
section_flow_data = df4.iloc[:, 1].tolist()

# 筛选“是”作为交路起终点的车站列表
start_end_stations = [index + 1 for index in df_station[df_station['是否可作为交路起点/终点'] == '是'].index.tolist()]
valid_combinations = []
# print(start_end_stations)
for sa in start_end_stations:
    for sb in start_end_stations:
        if sb - sa >= 3 and sb - sa <= 24:
            valid_combinations.append((sa, sb))

# 将相关的计算使用Numpy进行向量化
next_station_time = df_block['区间运行时间/s'].values
next_station_distance = df_block['站间距/km'].values

# 读取附件3 OD客流数据，并计算六类乘客数量以及乘客总的等待时间
file_path_OD = '附件3：OD客流数据.xlsx'
df_OD = pd.read_excel(file_path_OD, index_col=0).fillna(0)

# 读取附件4断面客流数据
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
sta_passenger_board = station_totals["上车总人数"].tolist()
sta_passenger_alight = station_totals["下车总人数"].tolist()

# print("下车总人数", sta_passenger_alight)

# 全局变量，用于记录目标函数的调用次数
objective_function_call_count = 0

def objective_function(x):
    # X = [ Nl, Ns, Sa, Sb, h, dwell_time*30 (6-35), Llb*30(36-65), Lla*30(66-95), Slb*30(96-125), Sla*30(126-155)]
    # 解析参数列表
    # 使用NumPy进行向量化操作，提高性能
    global objective_function_call_count
    objective_function_call_count += 1

    Bigloop_Train_num = x[0]
    smallloop_Train_num = x[1]
    Sa = int(x[2])  # 将x[2]强制转换为整数
    Sb = int(x[3])  # 将x[3]强制转换为整数
    headway = x[4]
    dwell_time = x[5:34]
    Llb = x[35:64]
    Lla = x[65:94]
    Slb = x[95:124]
    Sla = x[125:154]

    od_data = df_OD

    # 获取车站数量
    num_stations = len(next_station_distance) + 1

    # 生成车站名称列表，使用OD数据的索引
    station_names = od_data.index.tolist()

    # 初始化距离和时间矩阵的DataFrame，并设置行列索引
    passenger_time_matrix = pd.DataFrame(0, index=station_names, columns=station_names)
    distance_matrix = pd.DataFrame(0, index=station_names, columns=station_names)

    # 计算距离和时间矩阵，加上列车在车站的停站时间
    for i in range(num_stations - 1):
        for j in range(i + 1, num_stations):
            distance = sum(next_station_distance[i:j])
            time = sum(next_station_time[i:j]) + sum(dwell_time[i + 1:j])
            start_station = station_names[i]
            end_station = station_names[j]
            passenger_time_matrix.loc[start_station, end_station] = time
            distance_matrix.loc[start_station, end_station] = distance

    # 计算所有乘客总的在车时间
    passenger_invehicle_time = 0
    for start_station in od_data.index:
        for end_station in od_data.columns:
            passenger_count = od_data.at[start_station, end_station]
            passenger_invehicle_time += passenger_time_matrix.at[start_station, end_station] * passenger_count

    # 计算乘客类型数量和总的等待时间
    passenger_counts = {ptype: 0 for ptype in range(6)}  # 0到5代表不同的乘客类型

    for start_station in range(1, 31):
        for end_station in range(1, 31):
            if start_station < Sa and end_station < Sa:
                passenger_counts[0] += od_data.iloc[start_station - 1, end_station - 1]
            elif start_station < Sa and Sa <= end_station <= Sb:
                passenger_counts[1] += od_data.iloc[start_station - 1, end_station - 1]
            elif start_station < Sa and end_station > Sb:
                passenger_counts[2] += od_data.iloc[start_station - 1, end_station - 1]
            elif Sa <= start_station <= Sb and Sa <= end_station <= Sb:
                passenger_counts[3] += od_data.iloc[start_station - 1, end_station - 1]
            elif Sa <= start_station <= Sb and end_station > Sb:
                passenger_counts[4] += od_data.iloc[start_station - 1, end_station - 1]
            elif start_station > Sb and end_station > Sb:
                passenger_counts[5] += od_data.iloc[start_station - 1, end_station - 1]

    # 计算大交路的列车数与小交路的列车数的比例
    n = Bigloop_Train_num / smallloop_Train_num

    # 计算大交路的等待时间
    large_waiting_time = (n + 3) * headway / (2 * n + 2)

    # 计算小交路的等待时间
    small_waiting_time = headway / 2

    # 计算乘客总的等待时间
    passenger_waiting_time = (
        large_waiting_time * (passenger_counts[0] + passenger_counts[1] + passenger_counts[2] + passenger_counts[5]) +
        small_waiting_time * (passenger_counts[3] + passenger_counts[4]) + 0.5 * passenger_counts[4] * large_waiting_time
    )

    # 计算乘客的总在车时间和等待时间的总和（单位：秒）
    passenger_total_time = passenger_waiting_time + passenger_invehicle_time

    # 计算固定运营成本
    fixed_cost = Bigloop_Train_num + smallloop_Train_num

    # 计算灵活运营成本
    Bigloop_distance = distance_matrix.iloc[0, 29]  # 大交路的距离
    smallloop_distance = distance_matrix.iloc[Sa - 1, Sb - 1]  # 小交路的距离
    flexible_cost = Bigloop_Train_num * Bigloop_distance + smallloop_Train_num * smallloop_distance

    obj_function_res = passenger_total_time*0.008 + fixed_cost*40000000 + flexible_cost*1
    # 打印目标函数调用次数
    print(f"目标函数调用次数: {objective_function_call_count}")
    return obj_function_res


# x是一个含有155个元素的一卫列表，组成如上一行所示：
# 定义等式约束函数
def constraint_eq1(x):  # 判断(x[2], x[3])是否是valid_combinations中的元素
    small_l = (x[2], x[3])
    if small_l in valid_combinations:
        return 0
    else:
        print("eq1")
        return 1  # 返回-1表示违反了约束


# 定义等式约束函数
def constraint_eq2(x):
    total_violation = 0

    # 约束条件1
    for i in range(29):
        if i < x[2] or i >= x[3]:
            x[i + 95] = 0
        if i <= x[2] or i > x[3]:
            x[i + 125] = 0

    # 约束条件2
    for i in range(29):
        violation1 = sta_passenger_board[i] - x[i + 35] + x[i + 95]
        violation2 = sta_passenger_alight[i] - x[i + 65] + x[i + 125]

        # 计算约束违反程度，若不满足约束，则添加到总违反程度中
        total_violation += max(0, violation1, violation2)

    return total_violation


# 定义不等式约束函数
def constraint_ueq1(x):
    # 计算x[3] - x[2]的值
    diff = x[3] - x[2]
    # 检查是否满足条件：24 >= diff >= 3
    if 24 >= diff >= 3:
        return 0  # 满足约束
    else:
        return 1  # 违反约束


# 定义不等式约束函数
def constraint_ueq2(x):
    violations = []

    for item in valid_combinations:
        Sa, Sb = item
        x[2] = int(Sa)
        x[3] = int(Sb)
        if x[2] > 2:
            part1 = section_flow_data[0:int(x[2]) - 2]
            max_part1 = max(part1) if len(part1) > 0 else 0
        elif x[2] == 2:
            part1 = [section_flow_data[0]]
            max_part1 = max(part1) if len(part1) > 0 else 0
        else:
            max_part1 = 0

        if x[3] != 30:
            part3 = section_flow_data[int(x[3]) - 1:29]
            max_part3 = max(part3) if len(part3) > 0 else 0
        else:
            max_part3 = 0

        part2 = section_flow_data[int(x[2]) - 1:int(x[3]) - 2]
        max_part2 = max(part2) if len(part2) > 0 else 0

        violation1 = 1860 * x[0] - max_part1 + 1860 * x[0] - max_part3
        violation2 = 1860 * (x[0] + x[1]) - max_part2

        violation = violation1 + violation2
        violations.append(violation)

    return np.maximum(violations, 0).sum()


def constraint_ueq3(x):
    boarding_passenger_l = 0
    alight_passenger_l = 0
    boarding_passenger_s = 0
    alight_passenger_s = 0

    for i in range(1, 30):
        for j in range(1, 30):
            if j <= i:
                boarding_passenger_l += x[j - 1 + 35]
                alight_passenger_l += x[j - 1 + 65]
                boarding_passenger_s += x[j - 1 + 95]
                alight_passenger_s += x[j - 1 + 125]

    violation_l = max(boarding_passenger_l - alight_passenger_l - 1860, 0)
    violation_s = max(boarding_passenger_s - alight_passenger_s - 1860, 0)

    return violation_l + violation_s


# 定义不等式约束函数
def constraint_ueq4(x):
    total_violation = 0

    for i in range(29):
        if 0.04 * (x[i + 35] + x[i + 65]) > 0.04 * (x[i + 95] + x[i + 125]):
            result = 0.04 * (x[i + 35] + x[i + 65]) - x[i + 5]
        else:
            result = 0.04 * (x[i + 95] + x[i + 125]) - x[i + 5]

        violation = max(result, 0)
        total_violation += violation

    return total_violation


# 定义不等式约束函数
def constraint_ueq5(x):
    # 计算x[3] - x[2]的值
    train_num = x[0] + x[1]
    # 检查是否满足条件：24 >= diff >= 3
    if 30 >= train_num >= 10:
        return 0  # 满足约束
    else:
        # print("ueq5")
        return 1  # 违反约束


constraint_eq = [constraint_eq1, constraint_eq2]
constraint_ueq = [constraint_ueq1, constraint_ueq2, constraint_ueq3, constraint_ueq4, constraint_ueq5]

# 定义多目标遗传算法

size_pop = 40
max_iter = 300
ga = GA(func=objective_function, n_dim=155, size_pop=size_pop, max_iter=max_iter,
        lb=[1, 1, 1, 1, 120] + [20] * 30 + [1] * 30 + [1] * 30 + [1] * 30 + [1] * 30,
        ub=[30, 30, 30, 30, 360] + [120] * 30 + [1860] * 30 + [1860] * 30 + [1860] * 30 + [1860] * 30,
        constraint_eq=constraint_eq, constraint_ueq=constraint_ueq,
        precision=[1] * 155, prob_mut=0.1)

# 运行多目标遗传算法
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

Bigloop_Train_num =  best_x[0]
smallloop_Train_num =  best_x[1]
Sa = int(best_x[2])  # 将x[2]强制转换为整数
Sb = int(best_x[3])  # 将x[3]强制转换为整数
headway = best_x[4]
dwell_time = best_x[5:34]

print('Sa:',Sa)
print('Sb:',Sb)
print('headway:',headway)
print('dwell time:',dwell_time)

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()