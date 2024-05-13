"""
日期：2023.9.25
作者：CodeJohnw
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_train_schedules(train_data):
    """
    绘制多条列车的时间-距离曲线，显示它们从起始站到终点站的运行情况。

    Parameters:
        - train_data (list): 包含多组列车信息的列表，每组信息是一个字典，包含以下键：
            - 'departure_time' (int): 列车的出发时间（以秒为单位）。
            - 'start_station' (int): 列车的起始站。
            - 'end_station' (int): 列车的终点站。
            - 'dwell_time' (list): 列车在每个站点的停车时间列表（以秒为单位）。

    Returns:
        None
    """

    # 读取区间运行时间数据
    file_path = '附件2：区间运行时间.xlsx'
    df = pd.read_excel(file_path)

    # 提取区间运行时间和距离数据
    next_station_time = df['区间运行时间/s'].tolist()
    next_station_distance = df['站间距/km'].tolist()

    # 车站名称列表
    station_names = [f'Sta{i}' for i in range(1, 31)]

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制每组列车的时间距离曲线
    for train_info in train_data:
        departure_time = train_info['departure_time']
        start_station = train_info['start_station']
        end_station = train_info['end_station']
        dwell_time = train_info['dwell_time']

        # 初始化时间-距离数据点
        time = [0]  # 初始时间为0
        distance = [0]  # 初始距离为0
        current_distance = 0  # 初始距离为0
        current_time = departure_time # 设置初始时间为出发时间

        # 计算列车在各个车站的出发、停站、到达时间和距离
        for i in range(len(next_station_distance)):
            # 停站阶段
            current_time += dwell_time[i]  # 到达车站并停车的时间
            time.append(current_time)
            distance.append(current_distance)  # 在车站停车期间距离不变

            # 行驶期间
            current_time += next_station_time[i]  # 行驶到下一站的时间
            current_distance += next_station_distance[i]  # 更新距离
            time.append(current_time)
            distance.append(current_distance)

        # 提取要绘制的数据范围
        plot_distance = distance[2 * (start_station - 1):2 * (end_station)]
        # plot_time = [item - time[2 * (start_station - 1)] + departure_time for item in time[2 * (start_station - 1):2 * (end_station)]]
        plot_time = time[2 * (start_station - 1):2 * (end_station)]

        # 绘制列车的时间距离曲线，将linewidth设置为较小的值（例如0.5）
        plt.plot(plot_time, plot_distance, linestyle='-', label=f'Train Schedule ({start_station}-{end_station})', marker=None, linewidth=0.5)

    # 计算每个车站的纵轴位置
    station_positions = [sum(next_station_distance[:i]) for i in range(len(next_station_distance) + 1)]

    # 在每个车站的位置处标记车站名称
    for i in range(1, len(station_positions)):
        plt.axhline(y=station_positions[i], color='lightgrey', linestyle='--', linewidth=0.5)
        plt.text(0, station_positions[i], station_names[i], ha='right', va='center', color='black', fontsize=8)

    # 设置横轴的刻度线，时间范围为0点到3点
    plt.xticks(range(0, 10801, 1800), labels=[f'{i // 3600:02}:{(i % 3600) // 60:02}' for i in range(0, 10801, 1800)])
    ax.xaxis.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
    ax.xaxis.grid(True, which='minor', linestyle='--', linewidth='0.5', color='lightgray')

    # 添加半小时线
    ax.xaxis.set_minor_locator(plt.MultipleLocator(300))  # 300秒 = 5分钟
    ax.xaxis.set_major_locator(plt.MultipleLocator(1800))  # 1800秒 = 半小时

    # 设置横轴范围
    plt.xlim(0, 10800)  # 10800秒 = 3小时

    # 隐藏纵轴刻度标记，并将纵轴范围设置为0到最后一个车站的位置
    plt.yticks([])  # 隐藏纵轴刻度标记
    plt.ylim(0, station_positions[-1])  # 将纵轴范围设置为0到最后一个车站的位置

    # 显示图形
    plt.xlabel('Time')
    plt.ylabel('Station', labelpad=25)
    plt.title('Train Schedules')
    plt.grid(True, axis='y', linestyle='--', linewidth='0.5', color='lightgray')
    plt.legend()
    plt.show()



# 定义多组列车信息，每组信息是一个字典
train_info_1 = {
    'departure_time': 0,
    'start_station': 1,
    'end_station': 30,
    'dwell_time': [120] * 30
}

train_info_2 = {
    'departure_time': 900,
    'start_station': 5,
    'end_station': 23,
    'dwell_time': [120] * 30
}

train_info_3 = {
    'departure_time': 600,
    'start_station': 1,
    'end_station': 30,
    'dwell_time': [120] * 30
}

train_info_4 = {
    'departure_time': 300,
    'start_station': 1,
    'end_station': 30,
    'dwell_time': [120] * 30
}
train_info_5 = {
    'departure_time': 1200,
    'start_station': 1,
    'end_station': 30,
    'dwell_time': [120] * 30
}

train_info_6 = {
    'departure_time': 1500,
    'start_station': 1,
    'end_station': 30,
    'dwell_time': [120] * 30
}
train_info_7 = {
    'departure_time': 1800,
    'start_station': 5,
    'end_station': 23,
    'dwell_time': [120] * 30
}

# 调用绘图函数，传递多组列车信息
plot_train_schedules([train_info_1, train_info_2, train_info_3,train_info_4, train_info_5,train_info_6,train_info_7])
