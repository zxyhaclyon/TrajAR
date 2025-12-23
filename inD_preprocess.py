import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import pickle
import torch
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

INPUT_LENGTH = 96
PRED_HORIZON = 96
N_IN_FEATURES = 15
N_OUT_FEATURES = 15
DOWN_SAMPLE = 3
fz = 25

agent_dict = {'car':1, 'truck_bus':2, 'bicycle':3, 'pedestrian':4}

def create_directories(data_folder):
    root = f'./data/{data_folder}'
    top_dirs = ['training', 'validation', 'testing']
    sub_dirs = ['observation', 'target']
    for d in top_dirs:
        top_dir = f'{root}/{d}'
        if os.path.exists(top_dir):
            continue
        for s in sub_dirs:
            sub_dir = f'{top_dir}/{s}'
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


def find_neighboring_nodes(df, frame, id0, x0, y0, upper_limit=14):
    def filter_ids(dist, radius=50):
        return True if dist[0] < radius else False

    df1 = df[(df.frame == frame) & (df.trackId != id0)]
    if df1.empty:
        return []
    dist = list(df1.apply(lambda x: (euclidian(x0, y0, x.xCenter, x.yCenter), x.trackId), axis=1))
    dist = list(filter(filter_ids, dist))
    dist_sorted = sorted(dist)
    del dist_sorted[upper_limit:]
    return dist_sorted


def get_meta_property(tracks_meta, vehicle_ids, prop='class'):
    prp = [tracks_meta[tracks_meta.trackId == v_id][prop].values[0] for v_id in vehicle_ids]
    return prp


def wrap_to_pi(angle, deg2rad=True):
    if deg2rad:
        angle = np.deg2rad(angle)
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_input_features(df, frame_start, frame_end, trackId=-1):
    
    dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]
    x = list(map(lambda x:round(x,4), dfx.xCenter.values))[::DOWN_SAMPLE]
    y = list(map(lambda x:round(x,4), dfx.yCenter.values))[::DOWN_SAMPLE]
    vx = list(map(lambda x:round(x,4), dfx.xVelocity.values))[::DOWN_SAMPLE]
    vy = list(map(lambda x:round(x,4), dfx.yVelocity.values))[::DOWN_SAMPLE]
    ax = list(map(lambda x:round(x,4), dfx.xAcceleration.values))[::DOWN_SAMPLE]
    ay = list(map(lambda x:round(x,4), dfx.yAcceleration.values))[::DOWN_SAMPLE]
    light1 = [1 for i in range(len(x))]
    light2 = [1 for i in range(len(x))]
    light3 = [1 for i in range(len(x))]
    light4 = [1 for i in range(len(x))]
    light5 = [1 for i in range(len(x))]
    light6 = [1 for i in range(len(x))]
    light7 = [1 for i in range(len(x))]
    light8 = [1 for i in range(len(x))]
    return x, y, vx, vy, ax, ay, light1, light2, light3, light4, light5, light6, light7, light8

def get_target_features(df, frame_start, frame_end, n_features, agent_ltype, x0, y0):
    return_array = np.empty((32, n_features), dtype = np.float64)
    dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]

    x = list(map(lambda x:round(x,4), dfx.xCenter.values-x0))[::DOWN_SAMPLE]
    y = list(map(lambda x:round(x,4), dfx.yCenter.values-y0))[::DOWN_SAMPLE]
    vx = list(map(lambda x:round(x,4), dfx.xVelocity.values))[::DOWN_SAMPLE]
    vy = list(map(lambda x:round(x,4), dfx.yVelocity.values))[::DOWN_SAMPLE]
    ax = list(map(lambda x:round(x,4), dfx.xAcceleration.values))[::DOWN_SAMPLE]
    ay = list(map(lambda x:round(x,4), dfx.yAcceleration.values))[::DOWN_SAMPLE]
    light1 = [1 for i in range(len(x))]
    light2 = [1 for i in range(len(x))]
    light3 = [1 for i in range(len(x))]
    light4 = [1 for i in range(len(x))]
    light5 = [1 for i in range(len(x))]
    light6 = [1 for i in range(len(x))]
    light7 = [1 for i in range(len(x))]
    light8 = [1 for i in range(len(x))]
    
    feat_stack = np.stack((x, y, vx, vy, ax, ay, light1, light2, light3, light4, light5, \
                           light6, light7, light8, agent_ltype), axis=1)
  
    return_array[0:feat_stack.shape[0], :] = feat_stack
    return return_array


def get_adjusted_features(df, frame_start, frame_end, n_features, x0=0., y0=0., trackId=-1, nagent_type = -1):
    return_array = np.full((32, n_features), -1, dtype = np.float64)

    if trackId != -1:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end) & (df.trackId == trackId)]
    else:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]
    
    first_frame = dfx.frame.values[0]
    frame_offset = first_frame - frame_start
    padding = [-1 for i in range(frame_offset)]
    x = (padding + list(dfx.xCenter.values - x0))[::DOWN_SAMPLE]
    y = (padding + list(dfx.yCenter.values - y0))[::DOWN_SAMPLE]
    vx = (padding + list(dfx.xVelocity.values))[::DOWN_SAMPLE]
    vy = (padding + list(dfx.yVelocity.values))[::DOWN_SAMPLE]
    ax = (padding + list(dfx.xAcceleration.values))[::DOWN_SAMPLE]
    ay = (padding + list(dfx.yAcceleration.values))[::DOWN_SAMPLE]
    pad_num, true_num = x.count(-1), len(x) - x.count(-1)
    light1 = [-1 for i in range(pad_num)] + [1 for i in range(true_num)]
    light2 = [-1 for i in range(pad_num)] + [1 for i in range(true_num)]
    light3 = [-1 for i in range(pad_num)] + [1 for i in range(true_num)]
    light4 = [-1 for i in range(pad_num)] + [1 for i in range(true_num)]
    light5 = [-1 for i in range(pad_num)] + [1 for i in range(true_num)]
    light6 = [-1 for i in range(pad_num)] + [1 for i in range(true_num)]
    light7 = [-1 for i in range(pad_num)] + [1 for i in range(true_num)]
    light8 = [-1 for i in range(pad_num)] + [1 for i in range(true_num)]
    agent_ltype = [-1 for i in range(pad_num)] + [nagent_type for i in range(true_num)]

    feat_stack = np.stack((x, y, vx, vy, ax, ay, light1, light2, light3, light4, light5, \
                           light6, light7, light8, agent_ltype), axis=1)
    
    return_array[0:feat_stack.shape[0], :] = feat_stack
    return return_array


def get_storage_dict():
    dd = {}
    for t in ['training', 'validation', 'testing']:
        dd[t] = 0
    return dd


def remove_selected_vehicles(tracks, v_ids, rm=False):
    for v_id in v_ids:
        if rm:
            tracks = tracks.drop(tracks[(tracks.trackId == v_id)].index)
        else:
            tracks = tracks.drop(tracks[(tracks.trackId == v_id) &
                                        (tracks.xAcceleration == 0) &
                                        (tracks.yAcceleration == 0)].index)
    return tracks


def remove_parked_vehicles(tracks, tracks_meta):
    parked_vehicles = tracks_meta[(tracks_meta.initialFrame == 0) &
                                  (tracks_meta.finalFrame == tracks_meta.finalFrame.max())]
    a = parked_vehicles.trackId.values
    tracks = tracks[~tracks['trackId'].isin(a)]
    tracks_meta = tracks_meta[~tracks_meta['trackId'].isin(a)]
    return tracks, tracks_meta


def remove_still_vehicle(tracks, tracks_meta, exceptions=()):
    track_ids = pd.unique(tracks.trackId)
    a = []
    for ti in track_ids:
        if ti in exceptions:
            continue
        v_type = t_meta[t_meta.trackId == ti]['class'].iloc[0]
        if v_type in ('car', 'truck_bus'):
            dfx = tracks[tracks.trackId == ti]
            duration = dfx.trackLifetime.max()
            if duration > 2500:
                xc = dfx.xCenter.to_numpy()
                if np.all(xc == xc[0]):
                    a.append(ti)
    tracks = tracks[~tracks['trackId'].isin(a)]
    tracks_meta = tracks_meta[~tracks_meta['trackId'].isin(a)]
    return tracks, tracks_meta


def remove_parts(tracks, v_ids, duration):
    for vi, dur in zip(v_ids, duration):
        tracks = tracks.drop(tracks[(tracks.trackId == vi)
                                    & (tracks.trackLifetime > dur)].index)
    return tracks


def remove_pre_parts(tracks, v_ids, duration):
    for vi, dur in zip(v_ids, duration):
        tracks = tracks.drop(tracks[(tracks.trackId == vi)
                                    & (tracks.trackLifetime < dur)].index)
    return tracks

def euclidian_distance(x1, x2):
    # x1.shape (2, )
    # x2.shape (2, )
    return np.sqrt(np.sum((x1 - x2) ** 2))


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
    # if the frame starts in training set
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
    # if the frame starts in validation set
    if v_frames[0] >= val[0] and v_frames[0] <= val[-1]:
        if v_frames[-1] <= val[-1]:
            curr['validation'] = [0, v_frames[-1] - v_frames[0]]
        elif test[0] <= v_frames[-1] <= test[-1]:
            if (test[0] - v_frames[0]) >= time_len:
                curr['validation'] = [0, test[0] - v_frames[0] - 1]
            if (v_frames[-1] - test[0] + 1) >= time_len:
                curr['testing'] = [test[0] - v_frames[0], v_frames[-1] - v_frames[0]]
    # if the frame starts in test set
    if v_frames[0] >= test[0] and v_frames[-1] <= test[-1]:
        curr['testing'] = [0, v_frames[-1] - v_frames[0]]
    return curr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_save", default='inD-Location1', type=str, help="[data save path->inD-Location1/inD-Location2]")
    parser.add_argument("--data_read", default='./inD_data/location1', type=str, help="data read path")
    args = parser.parse_args()
    SEARCH_PATH = args.data_read
    data_folder = args.data_save
    data_root = create_directories(data_folder)
    # rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(18, 30)] # location2
    rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(7, 18)] # location1

    s_dict = get_storage_dict()
    np.random.seed(1234)
    for r_id in rec_ids:
        train_data, train_edge_feat, train_mask, train_target, train_tp_types = [], [], [], [], []
        valid_data, valid_edge_feat, valid_mask, valid_target, valid_tp_types = [], [], [], [], []
        test_data, test_edge_feat, test_mask, test_target, test_tp_types = [], [], [], [], []
        # only use some data
        if r_id in ('00', '01', '02', '03', '04', '05', '06'):
            p0 = (143.255269808385, -57.91170481615564)
        elif r_id in ('07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17'):
            p0 = (55.72110867398384, -32.74837088734138)
        elif r_id in ('18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'):
            p0 = (47.4118205383659, -28.8381176470473)
        else:
            p0 = (40.080060675120016, -25.416623842759034)
        print(f'Starting with recording {r_id}')
        meta = pd.read_csv(f'{SEARCH_PATH}/{r_id}_recordingMeta.csv')
        t_meta = pd.read_csv(f'{SEARCH_PATH}/{r_id}_tracksMeta.csv')
        tracks = pd.read_csv(f'{SEARCH_PATH}/{r_id}_tracks.csv', engine='pyarrow')
        # Perform some initial cleanup
        tracks, t_meta = remove_parked_vehicles(tracks, t_meta) # remove parked vehicles
        if r_id == '04':
            tracks, t_meta = remove_still_vehicle(tracks, t_meta, (141,)) # remove vehicles stopped for over 10 seconds，third element is Track-ID that exceeds time but moved
            tracks = remove_parts(tracks, (141,), (120,))
        elif r_id == '06':
            tracks, t_meta = remove_still_vehicle(tracks, t_meta, (0,))
            tracks = remove_pre_parts(tracks, (0,), (1700,))
        elif r_id == '24':
            tracks, t_meta = remove_still_vehicle(tracks, t_meta, (217,))
            tracks = remove_parts(tracks, (217,), (650,))
        elif r_id == '26':
            tracks, t_meta = remove_still_vehicle(tracks, t_meta, (31, 99))
            tracks = remove_parts(tracks, (31, 99), (630, 500))
        else:
            tracks, t_meta = remove_still_vehicle(tracks, t_meta)

        # Determine tr, val, test split (by frames)
        all_frame = max(tracks['frame'].values)
        train_frames, val_frames, test_frames = get_frame_split(all_frame) # according to照8:1:1split dataset（设置随机采样区间）

        # Get data and store
        all_ids = list(set(list(tracks['trackId'].values))) 
        ii = tqdm(range(0, len(all_ids)))
        for tri in ii:
            id0 = all_ids[tri]
            # Determine the vehicle type of the current trackId
            agent_type = t_meta[t_meta.trackId == id0]['class'].iloc[0]
            df = tracks[tracks.trackId == id0]
            frames = list(df.frame)
            # Filter out the data whose sequence lengths do not match
            if len(frames) < (INPUT_LENGTH + PRED_HORIZON):
                continue
            time_len = INPUT_LENGTH + PRED_HORIZON
            curr_split = which_set(frames, train_frames, val_frames, test_frames, time_len)
            if curr_split is None:
                # If a vehicle is within frames which are overlapping the sets
                continue
            for curr_set, indexl in curr_split.items():
                start_index = indexl[0]
                end_index =  indexl[-1]
                sub_frames = frames[start_index : end_index+1]
                for f in sub_frames[0:-time_len+1:fz*2]: # slide window for valid trajectories
                    fp = f + INPUT_LENGTH - 1
                    fT = fp + PRED_HORIZON
                    x, y, vx, vy, ax, ay, light1, light2, light3, \
                    light4, light5, light6, light7, light8 = get_input_features(df, f, fp) # features of historical window as input
                    agent_ltype = [agent_dict[agent_type] for i in range(len(x))]
                    neighbors = find_neighboring_nodes(tracks, fp, id0, x[-1], y[-1]) # select neighbors based on distance at last frame of history（可改进）
                    n_SVs = len(neighbors)
                    sv_ids = [int(neighbors[n][1]) for n in range(n_SVs)]
                    euc_dist = [int(neighbors[n][0]) for n in range(n_SVs)]
                    v_ids = [id0, *sv_ids]

                    x0 = p0[0]  # x[0]
                    y0 = p0[1]  # y[0]
                    # convert absolute coordinates to relative to origin
                    x = [round(xx - x0, 4) for xx in x]
                    y = [round(yy - y0, 4) for yy in y]
                    
                    # build input and output matrices
                    input_array = np.full((15, 32, N_IN_FEATURES), -1, dtype = np.float64) # determine input/output tensor dimensions
                    target_array = np.empty((32, N_OUT_FEATURES), dtype = np.float64)

                    input_array[0, :, :] = np.stack((x, y, vx, vy, ax, ay, light1, light2, light3,
                                                     light4, light5, light6, light7, light8, agent_ltype), axis=1)
                    
                    target_array[:, :] = get_target_features(df, fp + 1, fT, N_OUT_FEATURES, agent_ltype, x0, y0)
                    for j, n in enumerate(range(0, n_SVs)):
                        (dist, sv_id) = neighbors[n]
                        nagent_type = agent_dict[t_meta[t_meta.trackId == sv_id]['class'].iloc[0]]
                        input_array[j + 1, :, :] = get_adjusted_features(tracks, f, fp, N_IN_FEATURES, x0, y0, sv_id, nagent_type)

                    tp_type_array = np.zeros((15, 32, 3, 1))
                    for row in range(15):
                        for lie in range(int(32)):
                            if input_array[row, lie, -1] in [1,2]:
                                tp_type_array[row, lie] = np.array([[1], [0], [0]])
                            elif input_array[row, lie, -1] == 3:
                                tp_type_array[row, lie] = np.array([[0], [1], [0]])
                            elif input_array[row, lie, -1] == 4:
                                tp_type_array[row, lie] = np.array([[0], [0], [1]])
                    tp_type_array = torch.from_numpy(tp_type_array)

                    # Build edge features
                    input_edge_feat = euclidian_sequence(input_array) # (32, 15, 1)
     
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
        
        # each recording interval is treated as one file
        if not os.path.exists(f'./data/{data_root}/training/observation/{r_id}'):
            os.mkdir(f'./data/{data_root}/training/observation/{r_id}')
            os.mkdir(f'./data/{data_root}/training/target/{r_id}')
            os.mkdir(f'./data/{data_root}/validation/observation/{r_id}')
            os.mkdir(f'./data/{data_root}/validation/target/{r_id}')
            os.mkdir(f'./data/{data_root}/testing/observation/{r_id}')
            os.mkdir(f'./data/{data_root}/testing/target/{r_id}')
        torch.save(train_data, f'./data/{data_root}/training/observation/{r_id}/dat.pt')
        torch.save(train_mask, f'./data/{data_root}/training/observation/{r_id}/mask.pt')
        torch.save(train_edge_feat, f'./data/{data_root}/training/observation/{r_id}/edge_feat.pt')
        torch.save(train_tp_types, f'./data/{data_root}/training/observation/{r_id}/tp_types.pt')
        torch.save(train_target, f'./data/{data_root}/training/target/{r_id}/label.pt')
        
        torch.save(valid_data, f'./data/{data_root}/validation/observation/{r_id}/dat.pt')
        torch.save(valid_mask, f'./data/{data_root}/validation/observation/{r_id}/mask.pt')
        torch.save(valid_edge_feat, f'./data/{data_root}/validation/observation/{r_id}/edge_feat.pt')
        torch.save(valid_tp_types, f'./data/{data_root}/validation/observation/{r_id}/tp_types.pt')
        torch.save(valid_target, f'./data/{data_root}/validation/target/{r_id}/label.pt')

        torch.save(test_data, f'./data/{data_root}/testing/observation/{r_id}/dat.pt')
        torch.save(test_mask, f'./data/{data_root}/testing/observation/{r_id}/mask.pt')
        torch.save(test_edge_feat, f'./data/{data_root}/testing/observation/{r_id}/edge_feat.pt')
        torch.save(test_tp_types, f'./data/{data_root}/testing/observation/{r_id}/tp_types.pt')
        torch.save(test_target, f'./data/{data_root}/testing/target/{r_id}/label.pt')
        
        print(r_id, ' Training:', s_dict['training'], ' Validation:', s_dict['validation'], ' Testing:', s_dict['testing'])
