import random

# 创建一个名为 Train 的类
class Train:
    def __init__(self, departure_time, train_type):
        self.type = train_type  # 列车的类型
        self.passengers_num = 0  # 当前乘客数量，初始化为0
        self.passengers = [0] * 30  # 包含30个元素，表示乘客座位的列表
        self.arrive_time = [0]  # 包含一个元素，表示到达时间的列表，初始化为0
        self.depart_time = []  # 空列表，表示出发时间
        self.trans_passengers_num = 0  # 换乘乘客数量，初始化为0
        self.arrive_time.append(departure_time)  # 将传入的出发时间添加到到达时间列表中

# 创建一个名为 Station 的类
class Station:
    # 初始化 Station 类的实例
    def __init__(self, id, passengers_od, next_station_time, next_station_distance):
        self.id = id  # 站点的唯一标识符
        self.passenger_traffic = 0  # 乘客交通量，初始化为0
        self.passengers_od = passengers_od  # 乘客进出站的数据（可能需要初始化）
        self.passengers = 0  # 当前站点的乘客数量，初始化为0
        self.next_station_time = next_station_time  # 到达下一站所需的时间
        self.next_station_distance = next_station_distance  # 到达下一站的距离

# 创建一个名为 Simulator 的类
class Simulator:
    # 初始化 Simulator 类的实例
    def __init__(self, interval, scale, mini_start, mini_end):
        self.mini_start = mini_start  # Mini 的起始位置
        self.mini_end = mini_end  # Mini 的结束位置
        self.interval = interval  # 模拟的时间间隔
        self.scale = scale  # 缩放因子
        self.train_max_capacity = 1860  # 列车的最大容量
        self.safe_time_interval = 108  # 安全时间间隔
        self.clock = 0  # 模拟时钟
        self.train_departure_num = 0  # 列车出发数量
        self.train_last_departure = 0  # 上一次列车出发时间
        self.trains = []  # 列车列表
        self.stations = []  # 站点列表
        self.stations_init()  # 初始化站点

    def stations_init(self):
        # 在这里创建站点对象并添加到 self.stations 列表中
        for i in range(1, 31):
            station = Station(id=i, passengers_od={}, next_station_time=5, next_station_distance=10)
            self.stations.append(station)

    
    def simulate_train_run(self):
        for station in self.stations:
            # 随机生成该站点的乘客数量
            station.passengers = random.randint(0, 500)
            # 更新总的乘客交通量
            station.passenger_traffic += station.passengers
            # 更新列车上的乘客数量
            for train in self.trains:
                if train.type == "Passenger" and train.passengers_num < self.train_max_capacity:
                    passengers_boarding = min(station.passengers, self.train_max_capacity - train.passengers_num)
                    station.passengers -= passengers_boarding
                    train.passengers_num += passengers_boarding
                    # 更新列车的座位列表
                    for i in range(len(train.passengers)):
                        if train.passengers[i] == 0:
                            train.passengers[i] = 1
                            passengers_boarding -= 1
                            if passengers_boarding == 0:
                                break
            # 更新列车的出发时间
            for train in self.trains:
                if train.type == "Passenger" and train.passengers_num > 0:
                    train.depart_time.append(self.clock)
            # 输出站点和列车信息
            print(f"Clock: {self.clock}")
            for station in self.stations:
                print(f"Station {station.id}: Passengers={station.passengers}, Passenger Traffic={station.passenger_traffic}")
            for train in self.trains:
                print(f"Train Type: {train.type}, Passengers Num: {train.passengers_num}, Departure Time: {train.depart_time}")








# 示例用法
simulator = Simulator(interval=5, scale=1, mini_start=5, mini_end=25)
simulator.simulate_train_run()



######################################################计算乘客总的等待时间
def calculate_waiting_time_and_counts(Bigloop_Train_num, smallloop_Train_num, headway, Sa, Sb, od_data):
    '''
    计算乘客类型数量和总的等待时间
    
    Args:
    - Bigloop_Train_num (int): 大交路的列车数
    - smallloop_Train_num (int): 小交路的列车数
    - headway (int): 列车发车间隔时间（单位：分钟）
    - Sa (int): 小交路的起始车站编号
    - Sb (int): 小交路的终点车站编号
    - od_data (pd.DataFrame): OD客流数据，包含各站点之间的乘客数量
    
    Returns:
    - passenger_waiting_time (float): 乘客总等待时间（单位：分钟）
    - passenger_counts (dict): 各类乘客数量的字典，包括以下键值对：
        - "Class I Passenger" (int): I类乘客数量
        - "Class II Passenger" (int): II类乘客数量
        - "Class III Passenger" (int): III类乘客数量
        - "Class IV Passenger" (int): IV类乘客数量
        - "Class V Passenger" (int): V类乘客数量
        - "Class VI Passenger" (int): VI类乘客数量
    '''
    # 定义乘客类型的名称字典
    passenger_types = {
        0: "Class I Passenger",
        1: "Class II Passenger",
        2: "Class III Passenger",
        3: "Class IV Passenger",
        4: "Class V Passenger",
        5: "Class VI Passenger"
    }

    # 初始化各类乘客数量的字典
    passenger_counts = {ptype: 0 for ptype in passenger_types.values()}
    
    # 计算乘客类型数量
    for start_station in range(1, 31):
        for end_station in range(1, 31):
            if start_station < Sa and end_station < Sa:
                passenger_counts["Class I Passenger"] += od_data.iloc[start_station - 1, end_station - 1]
            elif start_station < Sa and Sa <= end_station <= Sb:
                passenger_counts["Class II Passenger"] += od_data.iloc[start_station - 1, end_station - 1]
            elif start_station < Sa and end_station > Sb:
                passenger_counts["Class III Passenger"] += od_data.iloc[start_station - 1, end_station - 1]
            elif Sa <= start_station <= Sb and Sa <= end_station <= Sb:
                passenger_counts["Class IV Passenger"] += od_data.iloc[start_station - 1, end_station - 1]
            elif Sa <= start_station <= Sb and end_station > Sb:
                passenger_counts["Class V Passenger"] += od_data.iloc[start_station - 1, end_station - 1]
            elif start_station > Sb and end_station > Sb:
                passenger_counts["Class VI Passenger"] += od_data.iloc[start_station - 1, end_station - 1]

    '''
    计算乘客总等待时间
    '''
    # 计算大交路的列车数与小交路的列车数的比例
    n = Bigloop_Train_num / smallloop_Train_num
    
    # 计算大交路的等待时间
    large_waiting_time = (n + 3) * headway / (2 * n + 2)
    
    # 计算小交路的等待时间
    small_waiting_time = headway / 2
    
    # 计算乘客总的等待时间
    passenger_waiting_time = (
    large_waiting_time * (passenger_counts["Class I Passenger"] + passenger_counts["Class II Passenger"] +
                         passenger_counts["Class III Passenger"] + passenger_counts["Class VI Passenger"]) +
    small_waiting_time * (passenger_counts["Class IV Passenger"] + passenger_counts["Class V Passenger"]) +
    0.5 * passenger_counts["Class V Passenger"] * large_waiting_time
    )
    return passenger_waiting_time, passenger_counts
    
    # # 示例用法
    # Sa = 5
    # Sb = 23
    # Bigloop_Trains = 10
    # smallloop_Trains = 2
    # headway = 5

    # total_waiting_time, passenger_counts = calculate_waiting_time_and_counts(Bigloop_Trains, smallloop_Trains, headway, Sa, Sb, df_OD)
    # print("乘客总等待时间:", total_waiting_time)
    # print("各类乘客数量:")
    # for ptype, count in passenger_counts.items():
    #     print(f"{ptype}: {count}")

######################################################计算乘客总的服务水平 = 乘客在车总时间 + 乘客等待总时间
def passenger_journey_time(passenger_waiting_time,passenger_invehicle_time):
    passenger_journey_time = passenger_waiting_time + passenger_invehicle_time
    return passenger_journey_time


######################################################计算企业运营成本
def operational_cost(Bigloop_Train_num, smallloop_Train_num, Sa, Sb, train_unit_price, distance_matrix):
    """
    计算列车运营成本

    参数:
    Bigloop_Train_num (int): 大交路列车数量
    smallloop_Train_num (int): 小交路列车数量
    Sa (int): 小交路起点站
    Sb (int): 小交路终点站
    train_unit_price (float): 单个列车固定成本
    distance_matrix (DataFrame): 列车运营距离矩阵

    返回值:
    total_operational_cost (float): 总运营成本
    """

    # 计算固定运营成本
    fixed_operational_cost = (Bigloop_Train_num + smallloop_Train_num) * train_unit_price

    # 计算灵活运营成本
    Bigloop_distance = distance_matrix.iloc[0, 29]  # 大交路的距离
    smallloop_distance = distance_matrix.iloc[Sa - 1, Sb - 1]  # 小交路的距离
    flexible_operational_cost = Bigloop_Train_num * Bigloop_distance + smallloop_Train_num * smallloop_distance

    # 计算总运营成本
    total_operational_cost = fixed_operational_cost + flexible_operational_cost

    return total_operational_cost



######################################################计算乘客总在车时间

def calculate_passenger_invehicle_time(dwell_time, od_data, next_station_time, next_station_distance):
    '''
    计算所有乘客总的在车时间、乘客在车时间矩阵和列车运营距离矩阵。

    参数：
    dwell_time (list): 1行30列的列表，包含列车在每个车站的停站时间（单位：秒）。
    od_data (DataFrame): OD客流数据，包含各站点之间的乘客数量。
    next_station_time (list): 区间运行时间列表，单位：秒。
    next_station_distance (list): 站间距离列表，单位：千米。

    返回值：
    total_passenger_time (float): 所有乘客总的在车时间（单位：秒）。
    passenger_time_matrix (DataFrame): 乘客在车时间矩阵，包含列车经过的车站停站时间。
    distance_matrix (DataFrame): 列车运营距离矩阵。
    '''

    # 获取车站数量
    num_stations = len(next_station_distance) + 1

    # 生成车站名称列表，使用od_data的索引
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

    return passenger_invehicle_time, passenger_time_matrix, distance_matrix