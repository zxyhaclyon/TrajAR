import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import torch
import os
import argparse

INPUT_LENGTH = 3.2
PRED_HORIZON = 3.2
N_IN_FEATURES = 15
N_OUT_FEATURES = 15
fz = 10
agent_dict = {'car':1, 'bus':2, 'truck':3, 'motorcycle':4, 'bicycle':5, 'tricycle':6, 'pedestrian':7}
normalize_dict = {'TianJin':{'x_min':-24.1609, 'x_max':55.5728, 'y_min':-8.9886, 'y_max':40.0838}, \
                  'ChangChun':{'x_min':-95.24867, 'x_max':54.0649, 'y_min':-82.8118, 'y_max':76.3586},\
                  'XiAn':{'x_min':-95.8782, 'x_max':75.5598, 'y_min':-20.9430, 'y_max':75.3351}, \
                  'ChongQing':{'x_min':-45.1188, 'x_max':51.0306, 'y_min':-26.4988, 'y_max':64.2722}
}

def create_directories(data_folder):
    root = f'./data/{data_folder}'
    top_dirs = ['training', 'validation', 'testing']
    sub_dirs = ['observation', 'target']
    exits_record = ['7_28_1','8_2_1','8_3_1','8_3_2','8_3_3','8_3_4','8_5_1','8_5_2','8_5_3','8_6_1','8_6_2','8_6_3', 
                    '8_6_4','8_7_1','8_7_2', '8_8_1', '8_9_1', '8_9_2', '8_9_3', '8_9_4','8_10_1','8_10_2','8_11_1']
    for d in top_dirs:
        top_dir = f'{root}/{d}'
        if os.path.exists(top_dir):  # and overwrite:
            continue
        for s in sub_dirs:
            for rid in exits_record:
                sub_dir = f'{top_dir}/{s}/{rid}'
                os.makedirs(sub_dir)
    return data_folder


def euclidian(x1, y1, x2, y2):
    from math import sqrt
    r = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return r


def maneuver_label(heading_start, heading_end):
    turn_alts = np.array([-np.pi / 2, 0, np.pi / 2, np.pi])
    tmp = heading_end - heading_start
    head_diff = turn_alts - np.radians(tmp)
    wrap_to_pi = np.arctan2(np.sin(head_diff), np.cos(head_diff))
    return np.argmin(np.abs(wrap_to_pi)), tmp


def find_neighboring_nodes(ped_df, veh_df, frame, id0, x0, y0, upper_limit=14):
    def filter_ids(sdist, radius=50):
        if sdist[0] < radius:
            return True
        else:
            return False

    df1 = ped_df[(ped_df.frame_id == frame) & (ped_df.track_id != id0)]
    df2 = veh_df[(veh_df.frame_id == frame) & (veh_df.track_id != id0)]
    if df1.empty and df2.empty:
        return []
    dist1, dist2 = [], []
    if not df1.empty:
        dist1 = list(df1.apply(lambda x: (euclidian(x0, y0, x.x, x.y), x.track_id), axis=1))
    if not df2.empty:
        dist2 = list(df2.apply(lambda x: (euclidian(x0, y0, x.x, x.y), x.track_id), axis=1))
    dist = dist1 + dist2
    dist = list(filter(filter_ids, dist))
    dist_sorted = sorted(dist)
    del dist_sorted[upper_limit:]
    return dist_sorted

def normalize_x(x):
    norm_dict = normalize_dict[City]
    x_min, x_max = norm_dict['x_min'], norm_dict['x_max']
    out = round(2 * (x - x_min) / (x_max - x_min) - 1, 4)
    return out

def normalize_y(y):
    norm_dict = normalize_dict[City]
    y_min, y_max = norm_dict['y_min'], norm_dict['y_max']
    out = round(2 * (y - y_min) / (y_max - y_min) - 1, 4)
    return out
    
def get_input_features(df, frame_start, frame_end):
    dfx = df[(df.frame_id >= frame_start) & (df.frame_id <= frame_end)]
    x = list(map(lambda x:round(x,4), dfx.x.values))
    y = list(map(lambda x:round(x,4), dfx.y.values))
    # x = list(map(normalize_x, x))
    # y = list(map(normalize_y, y))
    # 对xy坐标做归一化
    vx = list(map(lambda x:round(x,4), dfx.vx.values))
    vy = list(map(lambda x:round(x,4), dfx.vy.values))
    ax = list(map(lambda x:round(x,4), dfx.ax.values))
    ay = list(map(lambda x:round(x,4), dfx.ay.values))
    light1 = list(dfx.light1.values)
    light2 = list(dfx.light2.values)
    light3 = list(dfx.light3.values)
    light4 = list(dfx.light4.values)
    light5 = list(dfx.light5.values)
    light6 = list(dfx.light6.values)
    light7 = list(dfx.light7.values)
    light8 = list(dfx.light8.values)
    agent_type = list(map(lambda x:agent_dict[x], dfx.agent_type.values))
    return x, y, vx, vy, ax, ay, light1, light2, light3, light4, light5, light6, light7, light8, agent_type

def get_target_features(df, frame_start, frame_end, n_features):
    return_array = np.empty((frame_end - frame_start + 1, n_features), dtype = np.float64)
    dfx = df[(df.frame_id >= frame_start) & (df.frame_id <= frame_end)]
    
    first_frame = dfx.frame_id.values[0]
    frame_offset = first_frame - frame_start
    x = list(map(lambda x:round(x,4), dfx.x.values))
    y = list(map(lambda x:round(x,4), dfx.y.values))
    # x = list(map(normalize_x, x))
    # y = list(map(normalize_y, y))
    vx = list(map(lambda x:round(x,4), dfx.vx.values))
    vy = list(map(lambda x:round(x,4), dfx.vy.values))
    ax = list(map(lambda x:round(x,4), dfx.ax.values))
    ay = list(map(lambda x:round(x,4), dfx.ay.values))
    light1 = list(dfx.light1.values)
    light2 = list(dfx.light2.values)
    light3 = list(dfx.light3.values)
    light4 = list(dfx.light4.values)
    light5 = list(dfx.light5.values)
    light6 = list(dfx.light6.values)
    light7 = list(dfx.light7.values)
    light8 = list(dfx.light8.values)
    agent_type = list(map(lambda x:agent_dict[x], dfx.agent_type.values))
    
    feat_stack = np.stack((x, y, vx, vy, ax, ay, light1, light2, light3, light4, light5, \
                           light6, light7, light8, agent_type), axis=1)
    # 待预测节点和其邻居节点的轨迹长度不一，缺失的地方用NAN代替
    return_array[0:feat_stack.shape[0], :] = feat_stack
    return return_array

def get_adjusted_features(df, frame_start, frame_end, n_features, trackId = -1):
    return_array = np.full((frame_end - frame_start + 1, n_features), -1, dtype = np.float64)
    
    if trackId != -1:
        dfx = df[(df.frame_id >= frame_start) & (df.frame_id <= frame_end) & (df.track_id == trackId)]
    else:
        dfx = df[(df.frame_id >= frame_start) & (df.frame_id <= frame_end)]
    
    first_frame = dfx.frame_id.values[0]
    frame_offset = first_frame - frame_start
    x = list(map(lambda x:round(x,4), dfx.x.values))
    y = list(map(lambda x:round(x,4), dfx.y.values))
    # x = list(map(normalize_x, x))
    # y = list(map(normalize_y, y))
    vx = list(map(lambda x:round(x,4), dfx.vx.values))
    vy = list(map(lambda x:round(x,4), dfx.vy.values))
    ax = list(map(lambda x:round(x,4), dfx.ax.values))
    ay = list(map(lambda x:round(x,4), dfx.ay.values))
    light1 = list(dfx.light1.values)
    light2 = list(dfx.light2.values)
    light3 = list(dfx.light3.values)
    light4 = list(dfx.light4.values)
    light5 = list(dfx.light5.values)
    light6 = list(dfx.light6.values)
    light7 = list(dfx.light7.values)
    light8 = list(dfx.light8.values)
    agent_type = list(map(lambda x:agent_dict[x], dfx.agent_type.values))
    
    feat_stack = np.stack((x, y, vx, vy, ax, ay, light1, light2, light3, light4, light5, \
                           light6, light7, light8, agent_type), axis=1)
    # 待预测节点和其邻居节点的轨迹长度不一，缺失的地方用NAN代替
    return_array[frame_offset:frame_offset + feat_stack.shape[0], :] = feat_stack
    return return_array


def get_storage_dict():
    dd = {}
    for t in ['training', 'validation', 'testing']:
        dd[t] = 0
    return dd


def euclidian_distance(x1, x2):
    # x1.shape (2, )
    # x2.shape (2, )
    return round(np.sqrt(np.sum((x1 - x2) ** 2)), 2)


def euclidian_instance(inp):
    # inp.shape (n_vehicles, n_features)
    n_vehicles = inp.shape[0]
    output = []
    for v_id in range(0,1):
        for v_neighbor in range(n_vehicles):
            if inp[v_neighbor, 0] != -1:
                d = euclidian_distance(inp[v_id, :2], inp[v_neighbor, :2])
                output.append(d)
            else:
                output.append(-1)
    return torch.tensor(output).unsqueeze(1).numpy()


def euclidian_sequence(inp):
    # inp.shape (n_vehicles, seq_len, n_features)
    seq_len = inp.shape[1]
    output = []
    for i in range(0,seq_len):
        output.append(euclidian_instance(inp[:, i]))
    output = torch.from_numpy(np.array(output))
    return output


def get_frame_split(n_frames):
    all_frames = list(range(0, n_frames))
    # first variant 80-10-10
    tr = [0, all_frames[int(0.6 * n_frames) - 1]]
    val = [all_frames[int(0.6 * n_frames)], all_frames[int(0.8 * n_frames) - 1]]
    test = [all_frames[int(0.8 * n_frames)], all_frames[-1]]
    return tr, val, test


def which_set(v_frames, tr, val, test, time_len):
    assert v_frames[-1] > v_frames[0]
    curr = dict()
    # 如果在训练集阶段开始的frame
    if v_frames[0] >= tr[0] and v_frames[0] <= tr[-1]:
        if v_frames[-1] <= tr[-1]:
            curr['training'] = [0, v_frames[-1] - v_frames[0]]
        elif val[0] <= v_frames[-1] <= val[-1]:
            if (val[0] - v_frames[0]) >= time_len:
                curr['training'] = [0, val[0] - v_frames[0] - 1]
            if (v_frames[-1] - val[0] + 1) >= time_len:
                curr['validation'] = [val[0] - v_frames[0], v_frames[-1] - v_frames[0]]
        elif test[0] <= v_frames[-1] <= test[-1]:
            curr['validation'] = [val[0] - v_frames[0], val[-1] - v_frames[0]]
            if (val[0] - v_frames[0]) >= time_len:
                curr['training'] = [0, val[0] -v_frames[0] - 1]
            if (v_frames[-1] - test[0] + 1) >= time_len:
                curr['testing'] = [test[0] - v_frames[0], v_frames[-1] - v_frames[0]]
    # 如果在验证集阶段开始的frame
    if v_frames[0] >= val[0] and v_frames[0] <= val[-1]:
        if v_frames[-1] <= val[-1]:
            curr['validation'] = [0, v_frames[-1] - v_frames[0]]
        elif test[0] <= v_frames[-1] <= test[-1]:
            if (test[0] - v_frames[0]) >= time_len:
                curr['validation'] = [0, test[0] - v_frames[0] - 1]
            if (v_frames[-1] - test[0] + 1) >= time_len:
                curr['testing'] = [test[0] - v_frames[0], v_frames[-1] - v_frames[0]]
    # 如果在测试集阶段开始的frame
    if v_frames[0] >= test[0] and v_frames[-1] <= test[-1]:
        curr['testing'] = [0, v_frames[-1] - v_frames[0]]
    return curr

def city_traffic_light(trafficlight, column, all_timestamp, add_count):
    tra_dict = dict()
    ind = 0
    for ind in range(len(trafficlight)-1):
        trl1 = list(trafficlight.loc[ind][column])
        trl2 = list(trafficlight.loc[ind+1][column])
        if trl2[0] < 0:
            continue
        if trl1[0] < 0 and trl2[0] > 0:
            tra_dict[(0, trl2[0])] = list(trl1[1:]) * add_count
        if trl1[0] >= 0 and trl2[0] > 0:
            tra_dict[(trl1[0], trl2[0])] = list(trl1[1:]) * add_count
    # 加入最后时刻的信号灯状态
    trl1 = list(trafficlight.loc[ind+1][column])
    if all_timestamp > trl1[0]:
        tra_dict[(trl1[0], all_timestamp+1)] = list(trl1[1:]) * add_count
    return tra_dict

def create_traffic_light_dict(trafficlight, city, all_timestamp):
    feat_col = trafficlight.columns.tolist()
    column = ['timestamp(ms)']
    if city == 'Chongqing' or city == 'Tianjin':
        column.extend(list(feat_col[-8:]))
        tra_dict = city_traffic_light(trafficlight, column, all_timestamp, add_count = 1)
    elif city == 'Changchun' or city == 'Xian':
        column.extend(list(feat_col[-2:]))
        tra_dict = city_traffic_light(trafficlight, column, all_timestamp, add_count = 4)
    return tra_dict

def track_light_concat(tracks, traffic_light_dict, if_veh):
    origin_col = ['track_id','frame_id', 'timestamp_ms', 'agent_type','x','y','vx','vy','ax','ay']
    ori_tracks = tracks[origin_col]
    def match_timestamp(timestamp):
        for interval, values in traffic_light_dict.items():
            start, end = interval
            if start <= timestamp < end:
                result = values
                return result
        print(f'{timestamp}没有匹配上信号灯状态')
    # 应用函数到 DataFrame的每一行
    results = ori_tracks['timestamp_ms'].apply(match_timestamp)
    # 创建新的 8 个列
    light_columns = [f'light{i}' for i in range(1, 9)]
    ori_tracks[light_columns] = pd.DataFrame(results.tolist(), index=ori_tracks.index)
    return ori_tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--city", default='Tianjin', type=str, help="[Tianjin Xian]")
    parser.add_argument("--data_save", default='SinD-Tianjin', type=str, help="[data save path->SinD-Tianjin/SinD-Xian]")
    parser.add_argument("--data_read", default='./SinD_data/Tianjin', type=str, help="data read path")
    args = parser.parse_args()
    City = args.city
    data_folder = args.data_save
    data_root = create_directories(data_folder)
    # 指定要遍历的文件夹路径
    root_folder = args.data_read
    sub_folders = []
    # 使用os.walk()遍历文件夹
    for root, dirs, files in os.walk(root_folder):
        # 遍历子文件夹
        for dir in dirs:
            # 子文件夹的完整路径
            sub_folder_path = os.path.join(root_folder, dir)
            sub_folders.append(sub_folder_path)
    np.random.seed(1234)
    sub_folders = sorted(sub_folders)
    for sub_folder in sub_folders[:]:
        s_dict = get_storage_dict()
        train_data, train_edge_feat, train_mask, train_target, train_tp_types = [], [], [], [], []
        valid_data, valid_edge_feat, valid_mask, valid_target, valid_tp_types = [], [], [], [], []
        test_data, test_edge_feat, test_mask, test_target, test_tp_types = [], [], [], [], []
        record_id = list(sub_folder.split('/'))[-1]
        print(f'Starting with recording {record_id}')
        # 遍历文件夹中的数据集
        file_name = []
        for root, dirs, files in os.walk(sub_folder):
            for file in files:
                # 检查文件扩展名是否为.csv
                if file.endswith('.csv'):
                    file_name.append(file)
        ped_tracks, veh_tracks, trafficlight = 0, 0, 0
        for name in file_name:
            if name.startswith('Ped_smoothed_tracks'):
                ped_tracks = pd.read_csv(f'{sub_folder}/{name}')
            elif name.startswith('Veh_smoothed_tracks'):
                veh_tracks = pd.read_csv(f'{sub_folder}/{name}')
            elif name.startswith('Traffic'):
                trafficlight = pd.read_csv(f'{sub_folder}/{name}')
            else:
                if City != 'Tianjin':
                    print(f'{name}数据集匹配错误')
                    exit(0)
        all_frame = max([max(ped_tracks['frame_id'].values), max(veh_tracks['frame_id'].values)]) + 1
        all_timestamp = max([max(ped_tracks['timestamp_ms'].values), max(veh_tracks['timestamp_ms'].values)])
         # 生成交通信号灯的动态变化字典 键为时间区间左闭右开 值为信号灯的离散状态为float类型
        traffic_light_dict = create_traffic_light_dict(trafficlight, City, all_timestamp)
        # 将交通信号灯与轨迹做匹配
        ped_tracks = track_light_concat(ped_tracks, traffic_light_dict, if_veh = False)
        veh_tracks = track_light_concat(veh_tracks, traffic_light_dict, if_veh = True)
        # Determine tr, val, test split (by frames)
        train_frames, val_frames, test_frames = get_frame_split(all_frame) # 按照8:1:1划分数据集（设置随机采样区间）
        his_len = 32
        future_len = 32
        time_len = his_len + future_len
        # Get data and store
        veh_ids = set(list(veh_tracks['track_id'].values)) 
        ped_ids = set(list(ped_tracks['track_id'].values))
        agent_ids = list(veh_ids) + list(ped_ids)
        ii = tqdm(range(0, len(agent_ids)))
        for i in ii:
            id0 = agent_ids[i]
            if type(id0) == str:
                df = ped_tracks[ped_tracks.track_id == id0]
            else:
                df = veh_tracks[veh_tracks.track_id == id0]
            frames = list(df.frame_id)
            # 不满足时间窗口长度的轨迹删除
            if len(frames) < time_len: 
                continue
            # 轨迹划分数据集(可能长轨迹会划分进多个数据集)
            curr_split = which_set(frames, train_frames, val_frames, test_frames, time_len)
            if not curr_split:
                # If a vehicle is within frames which are overlapping the sets
                continue
            for curr_set, indexl in curr_split.items():
                start_index = indexl[0]
                end_index =  indexl[-1]
                sub_frames = frames[start_index : end_index+1]
                for f in sub_frames[0:-time_len+1:time_len]: # 对满足长度的车辆轨迹做滑窗处理
                    fp = f + his_len - 1
                    fT = fp + future_len
                    x, y, vx, vy, ax, ay, light1, light2, light3, \
                    light4, light5, light6, light7, light8, agent_type = get_input_features(df, f, fp) # 历史窗口的特征作为输入
                    neighbors = find_neighboring_nodes(ped_tracks, veh_tracks, fp, id0, x[-1], y[-1]) # 在历史窗口的最后一个帧根据距离选择邻居（可改进）
                    n_SVs = len(neighbors)
                    sv_ids = [neighbors[n][1] for n in range(n_SVs)]
                    euc_dist = [int(neighbors[n][0]) for n in range(n_SVs)]
                    v_ids = [id0, *sv_ids]

                    input_array = np.full((15, his_len, N_IN_FEATURES), -1, dtype = np.float64) # 输入输出向量维度确定
                    target_array = np.empty((future_len, N_OUT_FEATURES), dtype = np.float64)

                    input_array[0, :, :] = np.stack((x, y, vx, vy, ax, ay, light1, light2, light3,
                                                     light4, light5, light6, light7, light8, agent_type), axis=1)
                    target_array[:, :] = get_target_features(df, fp + 1, fT, N_OUT_FEATURES)
                    for j, n in enumerate(range(0, n_SVs)):
                        (dist, sv_id) = neighbors[n]
                        if type(sv_id) == str:
                            input_array[j + 1, :, :] = get_adjusted_features(ped_tracks, f, fp, N_IN_FEATURES, sv_id)
                        else:
                            input_array[j + 1, :, :] = get_adjusted_features(veh_tracks, f, fp, N_IN_FEATURES, sv_id)
                
                    # 为交通参与者的类型划分机动车、非机动车和行人
                    tp_type_array = np.zeros((15, his_len, 3, 1))
                    for row in range(15):
                        for lie in range(int(his_len)):
                            if input_array[row, lie, -1] in [1,2,3,4]:
                                tp_type_array[row, lie] = np.array([[1], [0], [0]])
                            elif input_array[row, lie, -1] in [5,6]:
                                tp_type_array[row, lie] = np.array([[0], [1], [0]])
                            elif input_array[row, lie, -1] == 7:
                                tp_type_array[row, lie] = np.array([[0], [0], [1]])
                    tp_type_array = torch.from_numpy(tp_type_array)
                    # Build edge indices 
                    # 为历史窗口的每一帧创建图结构，确定存在点在邻接矩阵中的索引关系
                    # input_edge_index = build_seq_edge_idx(torch.tensor(input_array)) # his_len * 2* (exit_node**2)
                    # Build edge features
                    input_edge_feat = euclidian_sequence(input_array) # (his_len, node, 1)

                    # Convert to torch tensors
                    input_array = torch.from_numpy(input_array).float()
                    target_array = torch.from_numpy(target_array).float()

                    # Compute masks
                    input_mask = (~torch.eq(input_array, -1)).to(torch.int)
                    input_mask_3d = (~torch.all(input_mask == 0, dim=-1)).unsqueeze(2).to(torch.int) # (node, his_len, 1)
                    
                    # 'training', 'validation', 'testing'
                    if curr_set == 'training':
                        train_data.append(input_array)
                        train_target.append(target_array)
                        train_edge_feat.append(input_edge_feat)
                        train_mask.append(input_mask_3d)
                        train_tp_types.append(tp_type_array)
                    elif curr_set == 'validation':
                        valid_data.append(input_array)
                        valid_target.append(target_array)
                        valid_edge_feat.append(input_edge_feat)
                        valid_mask.append(input_mask_3d)
                        valid_tp_types.append(tp_type_array)
                    elif curr_set == 'testing':
                        test_data.append(input_array)
                        test_target.append(target_array)
                        test_edge_feat.append(input_edge_feat)
                        test_mask.append(input_mask_3d)
                        test_tp_types.append(tp_type_array)

                    s_dict[curr_set] += 1
        
        # 一个城市的一段采集时间代表一个文件
        torch.save(train_data, f'./data/{data_root}/training/observation/{record_id}/dat.pt')
        torch.save(train_mask, f'./data/{data_root}/training/observation/{record_id}/mask.pt')
        torch.save(train_edge_feat, f'./data/{data_root}/training/observation/{record_id}/edge_feat.pt')
        torch.save(train_tp_types, f'./data/{data_root}/training/observation/{record_id}/tp_types.pt')
        torch.save(train_target, f'./data/{data_root}/training/target/{record_id}/label.pt')
        
        torch.save(valid_data, f'./data/{data_root}/validation/observation/{record_id}/dat.pt')
        torch.save(valid_mask, f'./data/{data_root}/validation/observation/{record_id}/mask.pt')
        torch.save(valid_edge_feat, f'./data/{data_root}/validation/observation/{record_id}/edge_feat.pt')
        torch.save(valid_tp_types, f'./data/{data_root}/validation/observation/{record_id}/tp_types.pt')
        torch.save(valid_target, f'./data/{data_root}/validation/target/{record_id}/label.pt')

        torch.save(test_data, f'./data/{data_root}/testing/observation/{record_id}/dat.pt')
        torch.save(test_mask, f'./data/{data_root}/testing/observation/{record_id}/mask.pt')
        torch.save(test_edge_feat, f'./data/{data_root}/testing/observation/{record_id}/edge_feat.pt')
        torch.save(test_tp_types, f'./data/{data_root}/testing/observation/{record_id}/tp_types.pt')
        torch.save(test_target, f'./data/{data_root}/testing/target/{record_id}/label.pt')
        
        print(record_id, ' Training:', s_dict['training'], ' Validation:', s_dict['validation'], ' Testing:', s_dict['testing'])
