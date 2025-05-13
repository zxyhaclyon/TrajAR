import argparse

def AddModelArgs(parser):

    parser.add_argument("--embed_scale", default=0.5 ,type=float)

    parser.add_argument("--input_dim", default=128 , type=int)

    parser.add_argument("--mtp_num_layers", default=1 , type=int)

    parser.add_argument("--mtp_num_heads", default=4 ,type=int)

    parser.add_argument("--mtp_feed_dim", default=256, type=int)
    
    parser.add_argument("--moe_topk", default=4, type=int)

    parser.add_argument("--num_shared_experts", default=3, type=int)
    
    parser.add_argument("--num_independent_experts", default=16, type=int)

    parser.add_argument("--moe_hid_dim", default=256 ,type=int)

    parser.add_argument("--moe_out_dim", default=256, type=int)

    parser.add_argument("--gat_out_dim", default=256, type=int)

    parser.add_argument("--gat_num_heads", default=4, type=int)

    parser.add_argument("--esa_num_layers", default=1, type=int)

    parser.add_argument("--esa_num_heads", default=4, type=int)

    parser.add_argument("--esa_feed_dim", default=512, type=int)
    
    parser.add_argument("--esa_dropout", default=0.1, type=float)

    parser.add_argument("--TrajAR_num_layers", default=6, type=int)

    parser.add_argument("--TrajAR_num_heads", default=4, type=int)

    parser.add_argument("--TrajAR_feed_dim", default=512, type=int)

    parser.add_argument("--TrajAR_dropout", default=0.1 ,type=float)

    parser.add_argument("--TrajAR_out_dim", default=256, type=int)


def AddDataArgs(parser):

    parser.add_argument("--data_root", default='./data/SinD-Xian', type=str)

    parser.add_argument("--dataset", default='SinD-Xian', type=str)

    parser.add_argument("--his_len",default=32, type=int)

    parser.add_argument("--predict_len", default=32, type=int)

    parser.add_argument("--small_ds", default=False, type=bool)

    parser.add_argument("--node_num",default=15, type=int)

    parser.add_argument("--dynamic_feat_num",default=6, type=int)

    parser.add_argument("--light_num", default=8, type=int)


def AddTrainArgs(parser):

    parser.add_argument("--lr", default=0.0001, type=float)

    parser.add_argument("--lr_decay", default=0.99, type=float)

    parser.add_argument("--weight_decay", default=0.05, type=float)

    parser.add_argument("--batch_size", default=32, type=int)

    parser.add_argument("--epoch", default=100, type=int)

    parser.add_argument("--val_epoch", default=1, type=int)

    parser.add_argument("--test_epoch", default=5, type=int)

    parser.add_argument("--patience", default=6, type=int)


def Xian_InitArgs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_root", default='./logs', type=str,
                                help="Log root directory")
    
    parser.add_argument("--nni" , default=False, type=bool)

    parser.add_argument("--save_result" , default=True, type=bool)
    
    parser.add_argument("--ctx" , default=2, type=int)

    AddDataArgs(parser)

    AddModelArgs(parser)

    AddTrainArgs(parser)

    args = parser.parse_args()

    return args