import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_tensor_type(torch.DoubleTensor)
from utils import check_dir
from utils import get_time_str
from datamodule import LitDataModule
from config.SinD_Tianjin_parser import Tianjin_InitArgs
from config.SinD_Xian_parser import Xian_InitArgs
from config.inD_Location1_parser import Location1_InitArgs
from config.inD_Location2_parser import Location2_InitArgs
from model import TrajAR
import copy
from torch.optim.lr_scheduler import ExponentialLR
import nni
import random
import string
import warnings
warnings.filterwarnings("ignore")
random_str = lambda : ''.join(random.sample(string.ascii_letters + string.digits, 6))

# 设置随机种子
randomSeed = 2046
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)

#-------------------------------------------------------------------------------------------------
def TrainEpoch(device, loader, model, optim, loss_fn, need_step, is_training):
    if need_step:
        model.train()
    else :
        model.eval()
        
    loss_item = 0
    count = 0
    for train_data in loader:
        x_dynamic = train_data.x_dynamic.to(device)
        x_light = train_data.x_light.to(device)
        x_agent_type = train_data.x_agent_type.to(device)
        y = train_data.y.to(device)
        graph_mask = train_data.graph_mask.to(device)
        graph_node_dis = train_data.graph_node_dis.to(device)
        tp_types = train_data.tp_types.to(device)
       
        loss = model(device, x_dynamic, x_light, x_agent_type, tp_types, y, graph_mask, graph_node_dis, is_training, loss_fn)
        loss_item += loss.item()
        count += 1
        
        if need_step:
            optim.zero_grad()
            loss.backward()
            optim.step()
  
    # 计算每个batch的平均损失
    loss_item /= count
    return loss_item

def ValidEpoch(device, loader, model, is_training):
    with torch.no_grad():
        model.eval()
        
        loss_item = 0
        count = 0
        for val_data in loader:
            x_dynamic = val_data.x_dynamic.to(device)
            x_light = val_data.x_light.to(device)
            x_agent_type = val_data.x_agent_type.to(device)
            y = val_data.y.to(device)
            graph_mask = val_data.graph_mask.to(device)
            graph_node_dis = val_data.graph_node_dis.to(device)
            tp_types = val_data.tp_types.to(device)
        
            predict = model(device, x_dynamic, x_light, x_agent_type, tp_types, y, graph_mask, graph_node_dis, is_training)
            target_xy = y[:,:,:2]
            xy_diff = predict - target_xy
            dist = torch.mean(torch.mean(torch.sqrt(torch.sum(xy_diff ** 2, dim=2)), dim=1))
            loss_item += dist.item()
            count += 1
        # 计算每个batch的平均损失
        loss_item /= count
        return loss_item

def TestEpoch(loader, model, save, is_training):
    with torch.no_grad():
        model.eval()
        targets = []
        predicts = []

        for test_data in loader:
            x_dynamic = test_data.x_dynamic.to(device)
            x_light = test_data.x_light.to(device)
            x_agent_type = test_data.x_agent_type.to(device)
            y = test_data.y.to(device)
            graph_mask = test_data.graph_mask.to(device)
            graph_node_dis = test_data.graph_node_dis.to(device)
            tp_types = test_data.tp_types.to(device)
            predict = model(device, x_dynamic, x_light, x_agent_type, tp_types, y, graph_mask, graph_node_dis, is_training)
            y = y[:,:,:2]

            predicts.append(list(predict.detach().cpu().numpy()))
            targets.append(list(y.detach().cpu().numpy()))

        ade_targets = np.array(targets) # (batch_num, bs, 32, 2)
        ade_predicts = np.array(predicts)
        fde_targets = np.array(targets)[:,:,-1,:] # (batch_num, bs, 2)
        fde_predicts = np.array(predicts)[:,:,-1,:]
        
        # 计算每个样本在每个时间步上预测值与目标值之间的欧几里得距离
        ade_distances = np.linalg.norm(ade_predicts - ade_targets, axis=-1)
        # 对每个样本在时间步维度上求平均，得到每个样本的ADE
        ade_per_sample = np.mean(ade_distances, axis=-1)
        # 对所有样本求平均，得到最终的ADE指标
        final_ade = np.mean(ade_per_sample)

        # 计算每个样本在每个时间步上预测值与目标值之间的欧几里得距离
        fde_distances = np.linalg.norm(fde_predicts - fde_targets, axis=-1)
        # 对每个样本在时间步维度上求平均，得到每个样本的ADE
        fde_per_sample = np.mean(fde_distances, axis=-1)
        # 对所有样本求平均，得到最终的ADE指标
        final_fde = np.mean(fde_per_sample)


    if save:
        np.savez(os.path.join(params_path,f'test.npz'), targets=targets, predicts=predicts)

    return final_ade, final_fde



def Train(args, model, device):

    patience_count = 0

    max_epoch = args.epoch

    lr = args.lr
    val_epoch = 1
    test_epoch = 1

    optim = torch.optim.AdamW(params=filter(lambda x : x.requires_grad, model.parameters()),lr=lr,weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer=optim, gamma=args.lr_decay)

    loss_fn = 'mse'

    best_loss = 1e9
    best_model = copy.deepcopy(model.grad_state_dict())

    train_loss_line = {'x':[],'y':[]}
    val_loss_line = {'x':[],'y':[]}

    for epoch in range(max_epoch):

        train_loss = TrainEpoch(device, train_loader, model, optim, loss_fn, need_step=True, is_training=True)

        train_loss_line['x'].append(epoch)
        train_loss_line['y'].append(train_loss)

        print(f"epoch {epoch} train_loss:{train_loss}")

        if epoch % val_epoch == 0:

            val_loss = ValidEpoch(device, valid_loader, model, is_training=False)
    
            val_loss_line['x'].append(epoch)
            val_loss_line['y'].append(val_loss)

            if val_loss < best_loss :
                patience_count = 0
                best_loss = val_loss
                best_model = copy.deepcopy(model.grad_state_dict())
            else :
                patience_count += 1
            
            if use_nni:
                nni.report_intermediate_result(val_loss)
            print(f"[Validation] epoch {epoch} val_loss:{val_loss}")

        if epoch % test_epoch == 0:

            ade, fde = TestEpoch(test_loader, model, save=False, is_training=False)

            print(f"[Test][prediction] epoch {epoch} ade:{ade} fde:{fde}")
        # 应该放在优化器更新参数之后，每个epoch更新学习率
        scheduler.step()
        print(f"[Scheduler] epoch {epoch} lr:{optim.param_groups[0]['lr']}")

        if patience_count >= args.patience:
                print('early stop')
                break

    # best_model = model.grad_state_dict()
    model.load_state_dict(best_model,strict=False)

    ade, fde = TestEpoch(test_loader, model, save=args.save_result, is_training=False)
    if use_nni:
        nni.report_final_result(ade)
    print(f"[Test][prediction] best model ade:{ade} fde:{fde}")  

if __name__ == '__main__':
    
    args = Tianjin_InitArgs() # 修改args函数读取不同数据集对应的参数
    use_nni = args.nni
    ctx = args.ctx
    device = torch.device(f'cuda:{ctx}' if torch.cuda.is_available() else "cpu")
    # 超参数设置
    if use_nni:
        params = nni.get_next_parameter()
        args.mtp_num_layers = params['mtp_num_layers']
        args.moe_topk = params['moe_topk']
        args.esa_num_layers = params['esa_num_layers']
        args.TrajAR_num_layers = params['TrajAR_num_layers']
        embed_scale = params['embed_scale']
        args.input_dim = int(embed_scale * args.input_dim)
        args.mtp_feed_dim = int(embed_scale * args.mtp_feed_dim)
        args.moe_hid_dim = int(embed_scale * args.moe_hid_dim)
        args.moe_out_dim = int(embed_scale * args.moe_out_dim)
        args.gat_out_dim = int(embed_scale * args.gat_out_dim)
        args.esa_feed_dim = int(embed_scale * args.esa_feed_dim)
        args.TrajAR_feed_dim = int(embed_scale * args.TrajAR_feed_dim)
        args.TrajAR_out_dim = int(embed_scale * args.TrajAR_out_dim)
    
    embed_scale = args.embed_scale
    args.input_dim = int(embed_scale * args.input_dim)
    args.mtp_feed_dim = int(embed_scale * args.mtp_feed_dim)
    args.moe_hid_dim = int(embed_scale * args.moe_hid_dim)
    args.moe_out_dim = int(embed_scale * args.moe_out_dim)
    args.gat_out_dim = int(embed_scale * args.gat_out_dim)
    args.esa_feed_dim = int(embed_scale * args.esa_feed_dim)
    args.TrajAR_feed_dim = int(embed_scale * args.TrajAR_feed_dim)
    args.TrajAR_out_dim = int(embed_scale * args.TrajAR_out_dim)
    
    #加载数据
    print('start load data')
    data_module = LitDataModule(args)
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print('End load data')
    #设置日志地址
    modelpath = ''
    if use_nni:
        params_path = args.model_root
        exp_id = nni.get_experiment_id()
        trail_id = nni.get_trial_id()
        param_path = str(exp_id) + '_' + str(trail_id)
        modelpath = os.path.join(params_path, f'{param_path}_model.pth')
    else:    
        params_path = os.path.join(args.model_root,f'{get_time_str()}_{args.dataset}_{random_str()}')
        check_dir(params_path,mkdir=True)
        modelpath = os.path.join(params_path,f'model.pth')

    #加载模型
    model = TrajAR(args).to(device)
    
    total_params, total_trainable_params = model.params_num()
    print(f'total_params:{total_params}    total_trainable_params:{total_trainable_params}')
    print('start training model')
    Train(args, model, device)

    #保存模型
    model.save(modelpath)



    



    
    
