from Data_Preparation import *
import numpy as np


# x是一个有155个元素的列表，其中元素如下X = [ Nl, Ns, Sa, Sb, h, dwell_time*30 (5-34), Llb*30(35-64), Lla*30(65-94), Slb*30(95-124), Sla*30(125-154)]  
'''
约束条件：



（1） (x[2],x[3])是 列表 valid_combinations = [(1, 5), (1, 8), (1, 10), (1, 14), (1, 17), (1, 18), (1, 21), (1, 22), (1, 25), (2, 5), (2, 8), (2, 10), (2, 14), (2, 17), (2, 18), (2, 21), (2, 22), (2, 25), (2, 26), (5, 8), (5, 10), (5, 14), (5, 17), (5, 18), (5, 21), (5, 22), (5, 25), (5, 26), (5, 27), (8, 14), (8, 17), (8, 18), (8, 21), (8, 22), (8, 25), (8, 26), (8, 27), (8, 30), (10, 14), (10, 17), (10, 18), (10, 21), (10, 22), (10, 25), (10, 26), (10, 27), (10, 30), (14, 17), (14, 18), (14, 21), (14, 22), (14, 25), (14, 26), (14, 27), (14, 30), (17, 21), (17, 22), (17, 25), (17, 26), (17, 27), (17, 30), (18, 21), (18, 22), (18, 25), (18, 26), (18, 27), (18, 30), (21, 25), (21, 26), (21, 27), (21, 30), (22, 25), (22, 26), (22, 27), (22, 30), (25, 30), (26, 30), (27, 30)]
中任意任一元组拆分的组合，即（x[2]，x[3]）in valid_combinations的某一元素
（2）24>=x[3] - x[2]>=3， 
（3）有一个包含 29 个元素的列表 section_flow_data，其中 x[2] 和 x[3] 分别用于划分列表 section_flow_data 成三个部分。part1 = section_flow_data[0:int(x[2]) - 2], part2 = section_flow_data[int(x[2]) - 1:int(x[3]) - 2],part3 = section_flow_data[int(x[3]) - 1:29]
对于第一个部分和第三个部分：
1860 * x[0] 应大于或等于这两个部分的最大值 Max(section_flow_data)，其中 i 为列表的索引。
对于第二个部分：
1860 * (x[0] + x[1]) 应大于或等于该部分的最大值 Max(section_flow_data)，其中 i 为列表的索引
(4)   x中的元素满足
    for i in range(29):
        if i < x[2] or i >= x[3]:
            x[i + 95] = 0
        if i <= x[2] or i > x[3]:
            x[i + 125] = 0

        sta_passenger_board[i] = x[i + 35] + x[i + 95]
        sta_passenger_alight[i] = x[i + 65] + x[i + 125]
sta_passenger_board 和sta_passenger_alight都是一个含有30个元素的列表

'''
#x是一个含有155个元素的一卫列表，组成如上一行所示：
# 定义等式约束函数
def constraint_eq1(x):## 判断(x[2], x[3])是否是valid_combinations中的元素
    small_l = (x[2], x[3])
    if small_l in valid_combinations:
        return 0
    else:
        return -1  # 返回-1表示违反了约束

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
        return -1  # 违反约束

# 定义不等式约束函数
# 示例：在约束函数中添加调试输出
def constraint_ueq2(x):
    violations = []
    # 切片section_flow_data
    for (x[2], x[3]) in valid_combinations:
        if x[2] == 1:
            max_part1 = 0
        else:
            part1 = section_flow_data[0:int(x[2]) - 2]
            max_part1 = max(part1)
        part2 = section_flow_data[int(x[2]) - 1:int(x[3]) - 2]
        part3 = section_flow_data[int(x[3]) - 1:29]

        # 计算各个部分的最大值
        
        max_part2 = max(part2)
        max_part3 = max(part3)

        # 计算约束违反程度
        violation1 = max(0, 1860 * x[0] - max_part1, 1860 * x[0] - max_part3)
        violation2 = max(0, 1860 * (x[0] + x[1]) - max_part2)

        # 返回约束违反程度的总和
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
        return -1  # 违反约束








constraint_eq = [constraint_eq1,constraint_eq2]
constraint_ueq = [constraint_ueq1, constraint_ueq2, constraint_ueq3, constraint_ueq4,constraint_ueq5]


