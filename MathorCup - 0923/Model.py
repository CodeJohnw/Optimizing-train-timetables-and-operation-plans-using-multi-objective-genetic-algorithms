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