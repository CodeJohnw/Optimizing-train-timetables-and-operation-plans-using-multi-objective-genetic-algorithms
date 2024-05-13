

import pandas as pd


def objective_function(x):
    '''
    计算目标函数值。

    Args:
        x (list): 包含列车数量、车站起始终点、停站时间等参数的列表。

    Returns:
        tuple: 包含三个目标函数值的元组，分别为总在车时间和等待时间的总和、固定运营成本、灵活运营成本。
    '''
    # X = [ Nl, Ns, Sa, Sb, h, dwell_time*30 (6-35), Llb*30(36-65), Lla*30(66-95), Slb*30(96-125), Sla*30(126-155)]  
    # 解析参数列表
    Bigloop_Train_num = x[0]
    smallloop_Train_num= x[1] 
    Sa = int(x[2])  # 将x[2]强制转换为整数
    Sb = int(x[3])  # 将x[3]强制转换为整数
    headway = x[4] 
    dwell_time = x[5:34] 
    Llb = x[35:64]
    Lla = x[65:94] 
    Slb = x[95:124] 
    Sla = x[125:154]

    # 读取区间运行时间和距离数据
    file_path_block = '附件2：区间运行时间.xlsx'
    df_block = pd.read_excel(file_path_block)

    # 获取区间运行时间和距离数据
    next_station_time = df_block['区间运行时间/s'].tolist()
    next_station_distance = df_block['站间距/km'].tolist()

    # 读取OD客流数据
    file_path_OD = '附件3：OD客流数据.xlsx'
    df_OD = pd.read_excel(file_path_OD, index_col=0).fillna(0)
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
            time = sum(next_station_time[i:j]) + sum(dwell_time[i+1:j])
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
    
    object_function = (passenger_total_time, fixed_cost, flexible_cost)

    return object_function
