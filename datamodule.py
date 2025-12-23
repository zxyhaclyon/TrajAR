import torch
import torch.utils.data as data
import numpy as np
import os

class LitDataModule():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.data_root = args.data_root
        self.small_ds = args.small_ds

    def train_dataloader(self):
        dataset = TrajectoryPredictionDataset('training',
                                              self.data_root,
                                              small=self.small_ds)
        print('training dataset batch number:', int(dataset.data_num / self.batch_size))
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last = True,
                          collate_fn=collate_session_based)

    def val_dataloader(self):
        dataset = TrajectoryPredictionDataset('validation',
                                              self.data_root,
                                              small=self.small_ds)
        print('validation dataset batch number:', int(dataset.data_num / self.batch_size))
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last = True,
                          collate_fn=collate_session_based)

    def test_dataloader(self):
        dataset = TrajectoryPredictionDataset('testing',
                                              self.data_root,
                                              small=self.small_ds)
        print('testing dataset batch number:', int(dataset.data_num / self.batch_size))
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=collate_session_based)


class TrajectoryPredictionDataset(data.Dataset):
    def __init__(self,
                 train_test: str,
                 data_set_src: str,
                 small: bool = False):
        self.mode = train_test
        self.root = data_set_src
        # Read the dataset in the folder
        obs_sub_folder = f'{self.root}/{self.mode}/observation'
        target_sub_folder = f'{self.root}/{self.mode}/target'
        input_data, nan_mask, node_adj, tp_types, target_data = [], [], [], [], [] 
        obs_sub_folders, target_sub_folders = [], []
        for root, dirs, files in os.walk(obs_sub_folder):
            for dir in dirs:
                sub_folder_path = os.path.join(obs_sub_folder, dir)
                obs_sub_folders.append(sub_folder_path)
        for root, dirs, files in os.walk(target_sub_folder):
            for dir in dirs:
                sub_folder_path = os.path.join(target_sub_folder, dir)
                target_sub_folders.append(sub_folder_path)
        
        for name in obs_sub_folders[:1]:
            inp = torch.load(f'{name}/dat.pt')
            input_data.extend(inp)
            mask = torch.load(f'{name}/mask.pt')
            nan_mask.extend(mask)
            edge_feat = torch.load(f'{name}/edge_feat.pt')  
            node_adj.extend(edge_feat)
            tp_type = torch.load(f'{name}/tp_types.pt')
            tp_types.extend(tp_type)
        for name in target_sub_folders:
            label = torch.load(f'{name}/label.pt')
            target_data.extend(label)
        
        self.input_data = torch.stack(input_data, dim = 0)
        self.nan_mask = torch.stack(nan_mask, dim = 0)
        self.node_distance = torch.stack(node_adj, dim = 0)
        self.target_data = torch.stack(target_data, dim = 0)
        self.tp_types = torch.stack(tp_types, dim = 0)
        self.data_num = self.input_data.shape[0]
        if small:
            # Smaller version for dry runs
            self.data_num = 50

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        """
        sample_input.shape (node_num, inp_seq_len, node_feats)
        sample_input_dynamic.shape (node_num, inp_seq_len, 6)
        sample_input_light.shape (inp_seq_len, 8, 1)
        sample_input_agent_type.shape (node_num, inp_seq_len, 1)
        sample_mask.shape (node_num, inp_seq_len, 1)
        sample_tp_types.shape (node_num, inp_seq_len, 3, 1)
        sample_target.shape (tar_seq_len, node_feats)
        sample_node_dis.shape (inp_seq_len, node_num, 1)
        """
        #  Model inputs
        sample_input = self.input_data[idx]
        sample_input_dynamic = sample_input[:,:,0:6].numpy()
        sample_input_light = sample_input[0,:,6:14].unsqueeze(-1).numpy()
        sample_input_agent_type = sample_input[:,:,-1].unsqueeze(-1).numpy()
        sample_mask = self.nan_mask[idx].numpy()
        sample_node_dis = self.node_distance[idx].numpy()
        sample_tp_types = self.tp_types[idx].numpy()

        #  Model targets
        sample_target = self.target_data[idx].numpy()

        return sample_input_dynamic, sample_input_light, sample_input_agent_type, sample_target, sample_mask, \
               sample_node_dis, sample_tp_types

def collate_session_based(batch):
    '''
    batch_x_dynamic.shape (batch_size, node_num, inp_seq_len, 6)
    batch_x_light.shape (batch_size, inp_seq_len, 8, 1)
    batch_x_agent_type.shape (batch_size, node_num, inp_seq_len, 1)
    batch_mask.shape (batch_size, node_num, inp_seq_len, 1)
    batch_target.shape (batch_size,tar_seq_len, node_feats)
    batch_graph_node_dis.shape (batch_size, inp_seq_len, node_num, 1)
    '''
    batch_size = len(batch)
    batch_x_dynamic = np.array([sample[0] for sample in batch])
    batch_x_light = np.array([sample[1] for sample in batch])
    batch_x_agent_type = np.array([sample[2] for sample in batch])
    batch_target = np.array([sample[3] for sample in batch])
    batch_mask = np.array([sample[4] for sample in batch])
    batch_graph_node_dis = np.array([sample[5] for sample in batch])
    batch_tp_types = np.array([sample[6] for sample in batch])
    return sample_data(batch_x_dynamic, batch_x_light, batch_x_agent_type, batch_target, batch_mask, \
                       batch_graph_node_dis, batch_tp_types)


class sample_data():
    def __init__(self, sample_input, sample_input_light, sample_input_agent_type, \
                 sample_target, sample_mask, sample_node_dis, sample_tp_types):
        self.x_dynamic = torch.from_numpy(sample_input).type(torch.DoubleTensor)
        self.x_light = torch.from_numpy(sample_input_light).type(torch.DoubleTensor)
        self.x_agent_type = torch.from_numpy(sample_input_agent_type).type(torch.LongTensor)
        self.y = torch.from_numpy(sample_target).type(torch.DoubleTensor)
        self.graph_mask = torch.from_numpy(sample_mask).type(torch.DoubleTensor)
        self.graph_node_dis = torch.from_numpy(sample_node_dis).type(torch.DoubleTensor)
        self.tp_types = torch.from_numpy(sample_tp_types).type(torch.DoubleTensor)

