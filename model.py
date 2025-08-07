import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from utils import cubic_spline_interpolation

# MOE Shared and Independent Experts
class SharedExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SharedExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class IndependentExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IndependentExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Mixture of Experts Model
class MOEModel(nn.Module):
    def __init__(self, top_k, num_shared_experts, num_independent_experts, input_dim, hidden_dim, output_dim):
        super(MOEModel, self).__init__()
        self.num_independent_experts = num_independent_experts
        self.output_dim = output_dim
        self.topk = top_k
        # Shared experts: one for each category (e.g., vehicle, non-vehicle, pedestrian)
        self.shared_experts = nn.ModuleList([SharedExpert(input_dim, hidden_dim, output_dim) for _ in range(num_shared_experts)])
        # Independent experts
        self.independent_experts = nn.ModuleList([IndependentExpert(input_dim, hidden_dim, output_dim) for _ in range(num_independent_experts)])
        # Gating network for independent experts
        self.gating_network = nn.Linear(input_dim, num_independent_experts)

    def forward(self, feature_vectors, object_types):
        '''
        feature_vectors: (batch_size, num_node, his_len, input_dim)
        object_types: (batch_size, num_node, his_len, 3, 1)
        '''
        batch_size, num_nodes, his_len, input_dim = feature_vectors.size()

        # Reshape input to (batch_size * num_nodes * his_len, input_dim)
        x_flat = feature_vectors.view(-1, input_dim)

        # Get independent experts gating scores
        gating_scores = self.gating_network(x_flat)  # (batch_size * num_nodes * his_len, num_independent_experts)
        # select top-k independent experts
        gating_scores = F.softmax(gating_scores, dim=1)
        indices = torch.topk(gating_scores, self.topk, dim=-1)[1]
        weights = gating_scores.gather(1, indices).type_as(feature_vectors)
        weights /= weights.sum(dim=-1, keepdim=True)
        
        # Select shared expert based on object type
        object_types = object_types.view(-1,3,1)  # (batch_size * num_nodes * his_len,3,1)
        # Generate a mask matrix to select shared experts (batch_size * num_nodes * his_len, 3, output_dim)
        obj_type_flat = object_types.repeat(1,1,self.output_dim).view(-1,3,self.output_dim)  
        # First, get the full output of the shared experts (batch_size * num_nodes * his_len, 3, output_dim)
        shared_expert_outputs = torch.stack([expert(x_flat) for expert in self.shared_experts], dim=1)
        # Then, select the corresponding shared expert output according to obj_type_flat (batch_size * num_nodes * his_len, output_dim)
        shared_expert_output = torch.mul(shared_expert_outputs,obj_type_flat).sum(dim=1)
        
        # Get independent expert outputs 
        independent_expert_output = torch.zeros_like(shared_expert_output)
        counts = torch.bincount(indices.flatten(), minlength=self.num_independent_experts)
        for i in range(self.num_independent_experts):
            if counts[i] == 0:
                continue
            expert = self.independent_experts[i]
            idx, top = torch.where(indices == i)
            independent_expert_output[idx] += expert(x_flat[idx]) * (weights[idx, top].unsqueeze(1))
        # Reshape output to (batch_size, num_nodes, his_len, output_dim)
        output = 0.5 * shared_expert_output + 0.5 * independent_expert_output
        output = output.view(batch_size, num_nodes, his_len, -1)
        return output
    
# Rotary Position Embedding
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, pos_dim):
        super(RotaryPositionEmbedding, self).__init__()
        self.pos_dim = pos_dim

    def forward(self, x, seq_len):
        # x: (batch, num_objects, seq_len, dim)

        # Generate position indices
        position = torch.arange(seq_len).unsqueeze(-1)  # (seq_len, 1)

        # Compute frequencies
        freqs = 10000 ** (-torch.arange(0, self.pos_dim, 2) / self.pos_dim)  # (dim // 2,)
        freqs = position * freqs.unsqueeze(0)  # (seq_len, dim // 2)

        # Compute sin and cos
        sin = torch.sin(freqs).to(device)  # (seq_len, dim // 2)
        cos = torch.cos(freqs).to(device)  # (seq_len, dim // 2)

        # Expand sin and cos to match input dimensions
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim // 2)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim // 2)

        # Apply rotary position embedding
        x_rotated = torch.empty(x.size()).to(device)
        x_rotated[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin  # Even indices
        x_rotated[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin  # Odd indices

        return x_rotated.to(device)
# Motion Trend Perception Transformer
class MTP_Mask_Attention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MTP_Mask_Attention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = int(d_model // nhead)

        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        '''
        x.shape(batch, num_node, his_len, d_model)
        mask.shape(batch, num_node, his_len, 1)
        '''
        batch_size, num_node, his_len, _ = x.size()
        x = x.view(batch_size * num_node, his_len, -1)
        # Linear projections
        Q = self.w_q(x)  # (batch_size * num_node, seq_len, d_model)
        K = self.w_k(x)    # (batch_size * num_node, seq_len, d_model)
        V = self.w_v(x)  # (batch_size * num_node, seq_len, d_model)

        # Split into multiple heads
        Q = Q.view(Q.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)  # (batch_size * num_node, nhead, seq_len, head_dim)
        K = K.view(K.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # Construct mask matrix (batch_size * num_node, nhead, seq_len, seq_len)
        mask2 = torch.transpose(mask, 2, 3)
        causal_mask = torch.matmul(mask,mask2)
        causal_mask = causal_mask.view(-1, his_len, his_len).unsqueeze(1)
        causal_mask = causal_mask.repeat(1, self.nhead, 1, 1)
        
        # Scaled Dot-Product Attention (batch_size * num_node, nhead, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  
        if mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))  # Apply mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size * num_node, nhead, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size * num_node, seq_len, d_model)

        # Final linear projection
        output = self.w_o(attn_output)  # (batch_size * num_node, seq_len, d_model)
        output = output.view(batch_size, num_node, his_len, -1)
        return output

class MTP_FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(MTP_FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MTP_TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super(MTP_TransformerEncoder, self).__init__()
        # Rotary position embedding
        self.rotary_pe = RotaryPositionEmbedding(input_dim)
        # Transformer encoder layers
        self.self_attn = MTP_Mask_Attention(input_dim, num_heads, dropout)
        self.feed_forward = MTP_FeedForward(input_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (batch, num_objects, seq_len, input_dim)
        batch_size, num_objects, seq_len, input_dim = x.shape

        # Apply rotary position embedding
        x = self.rotary_pe(x, seq_len)  # (batch, num_objects, seq_len, input_dim)

        # Pass through transformer encoder
        x = self.self_attn(x, mask)
        x = x + self.dropout1(x)
        x = self.norm1(x)
        x = self.feed_forward(x)
        x = x + self.dropout2(x)
        x = self.norm2(x)
        return x

# init_trajectory_encoder
class TrajFeatEncoder(nn.Module):
    def __init__(self, node_num, dynamic_feat_num, light_num, emb_dim):
        super().__init__()
        self.node_num = node_num
        self.dyfeat_enc = nn.Linear(in_features = dynamic_feat_num, out_features = emb_dim)
        self.light_enc = nn.Linear(in_features = 1, out_features = emb_dim)
        # self.light_enc = nn.Embedding(num_embeddings = 4, embedding_dim = emb_dim, padding_idx = 0)
        self.light_enc2 = nn.Linear(in_features = light_num * emb_dim, out_features = emb_dim)
        self.agent_enc = nn.Embedding(num_embeddings = 8, embedding_dim = emb_dim, padding_idx = 0)
    
    def forward(self, x_dynamic, x_light, x_agent_type):
        '''
        x_dynamic.shape (batch_size, node_num, inp_seq_len, 6) --> (batch_size, node_num, inp_seq_len, emb_dim)
        x_light.shape (batch_size, inp_seq_len, 8, 1) --> (batch_size, node_num, inp_seq_len, emb_dim)
        x_agent_type.shape (batch_size, node_num, inp_seq_len, 1) --> (batch_size, node_num, inp_seq_len, emb_dim)
        '''
        x_dynamic_out = self.dyfeat_enc(x_dynamic)
        # x_light = x_light.type(torch.LongTensor)
        x_light_enc = self.light_enc(x_light)
        x_light_enc = x_light_enc.view(x_light_enc.shape[0], x_light_enc.shape[1], -1)
        x_light_out = self.light_enc2(x_light_enc).unsqueeze(1)
        x_light_out = x_light_out.repeat(1, self.node_num, 1, 1)
        x_agent_out = self.agent_enc(torch.squeeze(x_agent_type))
        return x_dynamic_out, x_light_out, x_agent_out

class GAT_Model(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GAT_Model, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = int(out_dim // num_heads)
        
        # Linear transformation for node features
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(2*out_dim, out_dim)
        
        # Attention mechanism
        self.attn_acore = nn.Linear(2 * self.head_dim, 1)
        
    def forward(self, h, adj, mask):
        '''
        h: (batch_size, num_nodes, his_len, in_dim)
        adj: (batch_size, his_len, num_nodes, 1)
        mask: (batch_size, num_nodes, his_len, 1)
        '''
        batch_size, num_nodes, his_len, in_dim = h.size()
        
        # Linear transformation
        h = self.linear(h)  # (batch_size, num_nodes, his_len, out_dim)
        target_h = h[:, 0, :, :]
        # Reshape for multi-head attention
        h = h.view(batch_size, num_nodes, his_len, self.num_heads, self.head_dim)
        h = h.permute(0, 2, 3, 1, 4)  # (batch_size, his_len, num_heads, num_nodes, head_dim)
        
        # Compute attention scores
        h_src = h.unsqueeze(4)  # (batch_size, his_len, num_heads, num_nodes, 1, head_dim)
        h_dst = h.unsqueeze(3)  # (batch_size, his_len, num_heads, 1, num_nodes, head_dim)
        # (batch_size, his_len, num_heads, num_nodes, num_nodes, 2 * head_dim)
        h_cat = torch.cat([h_src.expand(-1, -1, -1, -1, num_nodes, -1),
                           h_dst.expand(-1, -1, -1, num_nodes, -1, -1)], dim=-1) 
        attn_scores = self.attn_acore(h_cat).squeeze(-1)  # (batch_size, his_len, num_heads, num_nodes, num_nodes)
        
        # Apply mask and softmax
        single_attn_score = attn_scores[:, :, :, 0, :] # (batch_size, his_len, num_heads, num_nodes)
        mask = mask.permute(0, 2, 3, 1).contiguous().repeat(1, 1, self.num_heads, 1) # (batch_size, his_len, num_heads, num_nodes)
        adj = adj.repeat(1, 1, 1, self.num_heads).permute(0, 1, 3, 2).contiguous() # (batch_size, his_len, num_heads, num_nodes)
        adj = adj.masked_fill(adj == 0, 1e9)
        adj = torch.where(mask>0, 1/adj, adj)
        single_attn_score = single_attn_score.masked_fill(mask == 0, -1e9)
        attn_scores = F.softmax(single_attn_score, dim=-1)
        attn_scores = torch.mul(attn_scores, adj)
        attn_scores = attn_scores.unsqueeze(3) # (batch_size, his_len, num_heads, 1, num_nodes)
        # Apply attention
        h = torch.matmul(attn_scores, h).squeeze(3)  # (batch_size, his_len, num_heads, head_dim)
        
        # Concatenate heads and reshape
        h = h.view(batch_size, his_len, self.out_dim)  # (batch_size, his_len, out_dim)
        out_h = self.linear2(torch.cat([h, target_h], dim=-1))
        return out_h

class ESA_EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(ESA_EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src):
        # src: (batch, his_len, d_model)
        src = src.transpose(0, 1)  # (his_len, batch, d_model) for MultiheadAttention
        src = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = src + self.dropout2(src2)
        src2 = self.norm2(src2)

        src2 = src2.transpose(0, 1)  # (batch, his_len, emb)
        return src2

class ESA_TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        super(ESA_TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([ESA_EncoderLayer(d_model, nhead, dim_feedforward, dropout) \
                                     for _ in range(num_layers)])
    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

class TrajAR_decoder_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TrajAR_decoder_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.out_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Split into multiple heads
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Final linear layer
        output = self.out_linear(attn_output)
        
        return output   

class TrajAR_Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TrajAR_Decoder, self).__init__()
        self.self_attn = TrajAR_decoder_Attention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # x: (batch_size, his_len+61, embed_dim)
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class TrajAR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.node_num = args.node_num
        self.dynamic_feat_num = args.dynamic_feat_num
        self.light_num = args.light_num
        self.input_dim = args.input_dim
        self.mtp_num_layers = args.mtp_num_layers
        self.mtp_num_heads = args.mtp_num_heads
        self.mtp_feed_dim = args.mtp_feed_dim
        self.moe_topk = args.moe_topk
        self.num_shared_experts = args.num_shared_experts
        self.num_independent_experts = args.num_independent_experts
        self.moe_hid_dim = args.moe_hid_dim
        self.moe_out_dim = args.moe_out_dim
        self.gat_out_dim = args.gat_out_dim
        self.gat_num_heads = args.gat_num_heads
        self.esa_num_layers = args.esa_num_layers
        self.esa_num_heads = args.esa_num_heads
        self.esa_feed_dim = args.esa_feed_dim
        self.esa_dropout = args.esa_dropout
        self.TrajAR_num_layers = args.TrajAR_num_layers
        self.TrajAR_num_heads = args.TrajAR_num_heads
        self.TrajAR_feed_dim = args.TrajAR_feed_dim
        self.TrajAR_dropout = args.TrajAR_dropout
        self.TrajAR_out_dim = args.TrajAR_out_dim
        # Trajectory initial feature encoder
        self.init_traj_enc = TrajFeatEncoder(self.node_num, self.dynamic_feat_num, self.light_num, self.input_dim)
        # Motion Trend Perception Transformer
        self.mtp_transformer = nn.ModuleList([MTP_TransformerEncoder(self.input_dim, self.mtp_feed_dim, self.mtp_num_heads) \
                                              for _ in range(self.mtp_num_layers)])
        # MOE model
        self.moe_model = MOEModel(top_k=self.moe_topk, num_shared_experts=self.num_shared_experts, num_independent_experts=self.num_independent_experts, \
                                  input_dim=self.input_dim, hidden_dim=self.moe_hid_dim, output_dim=self.moe_out_dim)
    
        # GAT model (batch_size, his_len, gat_out_dim)
        self.gatlayer = GAT_Model(self.moe_out_dim, self.gat_out_dim, self.gat_num_heads)

        # ESA-Transformermox (batch_size, his_len, gat_out_dim)
        self.esa_model = ESA_TransformerEncoder(self.gat_out_dim, self.esa_num_heads, self.esa_feed_dim, \
                                                self.esa_dropout, self.esa_num_layers)
        # Residual connection encoder
        self.esaout_resnet = nn.Linear(in_features=self.input_dim, out_features=self.gat_out_dim)
        
        # TrajAR-Trajectory-embedding
        self.traj_embed = nn.Linear(2, self.gat_out_dim)
        self.final_traj_embed = nn.Linear(int(args.his_len), 1)
        self.TrajAR = nn.ModuleList([
            TrajAR_Decoder(self.gat_out_dim, self.TrajAR_num_heads, self.TrajAR_feed_dim, self.TrajAR_dropout) \
            for _ in range(self.TrajAR_num_layers)
        ])
        self.out_linear2 = nn.Linear(self.TrajAR_out_dim, 2)
        
    def forward(self, my_device, x_dynamic, x_light, x_agent_type, object_types, y_targte, graph_mask, graph_node_dis, is_training, loss_func=None):
        '''
        x_dynamic.shape (batch_size, node_num, inp_seq_len, 6)
        x_light.shape (batch_size, inp_seq_len, 8, 1)
        x_agent_type.shape (batch_size, node_num, inp_seq_len, 1)
        object_types.shape (batch_size, node_num, inp_seq_len, 3, 1)
        graph_mask.shape (batch_size, node_num, inp_seq_len, 1)
        y_targte.shape (batch_size, tar_seq_len, node_feats)
        graph_node_dis.shape (batch_size, inp_seq_len, node_num, 1)
        '''
        batch_size, num_node, inp_his_len, _ = x_dynamic.size()
        global device
        device = my_device
       
        # Encode trajectory points (batch_size, node_num, inp_seq_len, input_dim)
        x_agent_type = torch.mul(x_agent_type, graph_mask).type(torch.LongTensor).to(device)
        x_dynamic_enc, x_light_enc, x_agent_enc = self.init_traj_enc(x_dynamic, x_light, x_agent_type)
        origin_x_dynamic = x_dynamic_enc.clone()
        x_dynamic_ego = x_dynamic_enc[:, 0, :, :].to(device)
       
        # Transformer encodes the position and dynamic features of the trajectory points (batch_size, node_num, inp_seq_len, input_dim)
        for mtp_model in self.mtp_transformer:
            x_dynamic_enc = mtp_model(x_dynamic_enc, graph_mask)
       
        # Fuse trajectory features, signal light features, and traffic participant type features
        fuse_x = x_dynamic_enc + x_light_enc + x_agent_enc
        
        # MOE models priority representation (batch_size, num_nodes, his_len, moe_out_dim)
        moe_output = self.moe_model(fuse_x, object_types)
       
        # GAT models spatial correlation (batch_size, his_len, gat_out_dim)
        gat_output = self.gatlayer(moe_output, graph_node_dis, graph_mask)
       
        # ESA-Transformer models environment perception (batch_size, his_len, gat_out_dim)
        esa_output = self.esa_model(gat_output)
       
        # Residual connection
        esa_output = self.esaout_resnet(x_dynamic_ego) + esa_output
        
        # Train TrajAR
        final_future_embed = self.final_traj_embed(esa_output.permute(0,2,1)).contiguous()
        final_future_embed = final_future_embed.view(batch_size,1,self.gat_out_dim)
        # First autoregression, generating the last trajectory point
        padding_embed = torch.zeros((batch_size, 62, self.gat_out_dim), device = x_dynamic.device, dtype=torch.float64)
        AR_embed1 = torch.cat([esa_output, final_future_embed, padding_embed], dim = 1)
        AR_causal_mask1 = torch.zeros((batch_size, inp_his_len+63, inp_his_len+63), device = x_dynamic.device, dtype=torch.float64)
        AR_causal_mask1[:, inp_his_len:inp_his_len+1, :inp_his_len+1] = 1
        TrajAR_out1 = AR_embed1
        for trajAR_model in self.TrajAR:
            TrajAR_out1 = trajAR_model(TrajAR_out1, AR_causal_mask1)
        out_traj_xy = self.out_linear2(TrajAR_out1)
        pred_xy1 = out_traj_xy[:,inp_his_len:inp_his_len+1,:]
        pred_xy1_embed = TrajAR_out1[:,inp_his_len:inp_his_len+1,:] # 维度没有改变
        # Second autoregression, generating the 16th and 32nd trajectory points
        pre_interpolation_feat1 = torch.stack([x_dynamic[:, 0, 0, :2], x_dynamic[:, 0, -1, :2], pred_xy1.squeeze()], dim = 1)
        interpolation_mat1 = cubic_spline_interpolation(pre_interpolation_feat1, 1)[:, 3:, :]
        interpolation_embed1 = self.traj_embed(interpolation_mat1)
        padding_embed2 = torch.zeros((batch_size, 60, self.gat_out_dim), device = x_dynamic.device, dtype=torch.float64)
        AR_embed2 = torch.cat([esa_output, pred_xy1_embed, interpolation_embed1, padding_embed2], dim = 1)
        AR_causal_mask2 = torch.zeros((batch_size, inp_his_len+63, inp_his_len+63), device = x_dynamic.device, dtype=torch.float64)
        AR_causal_mask2[:, inp_his_len+1:inp_his_len+3, :inp_his_len+3] = 1
        TrajAR_out2 = AR_embed2
        for trajAR_model in self.TrajAR:
            TrajAR_out2 = trajAR_model(TrajAR_out2, AR_causal_mask2)
        pred_xy2_embed = TrajAR_out2[:,inp_his_len+1:inp_his_len+3,:]
        out_traj_xy2 = self.out_linear2(TrajAR_out2)
        pred_xy2 = out_traj_xy2[:,inp_his_len+1:inp_his_len+3,:]
        # Third autoregression, generating the 8th, 16th, 24th, and 32nd trajectory points
        pre_interpolation_feat2 = torch.cat([x_dynamic[:, 0, -1, :2].unsqueeze(1), pred_xy2], dim = 1)
        interpolation_mat2 = cubic_spline_interpolation(pre_interpolation_feat2, 1)[:, 1:, :]
        interpolation_embed2 = self.traj_embed(interpolation_mat2)
        padding_embed3 = torch.zeros((batch_size, 56, self.gat_out_dim), device = x_dynamic.device, dtype=torch.float64)
        AR_embed3 = torch.cat([esa_output, pred_xy1_embed, pred_xy2_embed, interpolation_embed2, padding_embed3], dim = 1)
        AR_causal_mask3 = torch.zeros((batch_size, inp_his_len+63, inp_his_len+63), device = x_dynamic.device, dtype=torch.float64)
        AR_causal_mask3[:, inp_his_len+3:inp_his_len+7, :inp_his_len+7] = 1
        TrajAR_out3 = AR_embed3
        for trajAR_model in self.TrajAR:
            TrajAR_out3 = trajAR_model(TrajAR_out3, AR_causal_mask3)
        pred_xy3_embed = TrajAR_out3[:,inp_his_len+3:inp_his_len+7,:]
        out_traj_xy3 = self.out_linear2(TrajAR_out3)
        pred_xy3 = out_traj_xy3[:,inp_his_len+3:inp_his_len+7,:]
        # Fourth autoregression, generating the 4th, 8th, 12th, 16th, 20th, 24th, 28th, and 32nd trajectory points
        pre_interpolation_feat3 = torch.cat([x_dynamic[:, 0, -1, :2].unsqueeze(1), pred_xy3], dim = 1)
        interpolation_mat3 = cubic_spline_interpolation(pre_interpolation_feat3, 1)[:, 1:, :]
        interpolation_embed3 = self.traj_embed(interpolation_mat3)
        padding_embed4 = torch.zeros((batch_size, 48, self.gat_out_dim), device = x_dynamic.device, dtype=torch.float64)
        AR_embed4 = torch.cat([esa_output, pred_xy1_embed, pred_xy2_embed, pred_xy3_embed, interpolation_embed3, padding_embed4], dim = 1)
        AR_causal_mask4 = torch.zeros((batch_size, inp_his_len+63, inp_his_len+63), device = x_dynamic.device, dtype=torch.float64)
        AR_causal_mask4[:, inp_his_len+7:inp_his_len+15, :inp_his_len+15] = 1
        TrajAR_out4 = AR_embed4
        for trajAR_model in self.TrajAR:
            TrajAR_out4 = trajAR_model(TrajAR_out4, AR_causal_mask4)
        pred_xy4_embed = TrajAR_out4[:,inp_his_len+7:inp_his_len+15,:]
        out_traj_xy4 = self.out_linear2(TrajAR_out4)
        pred_xy4 = out_traj_xy4[:,inp_his_len+7:inp_his_len+15,:]
        # Fifth autoregression, generating the 2nd, 4th, 6th, 8th, 10th, 12th, 14th, 16th, 18th, 20th, 22nd, 24th, 26th, 28th, and 30th trajectory points
        pre_interpolation_feat4 = torch.cat([x_dynamic[:, 0, -1, :2].unsqueeze(1), pred_xy4], dim = 1)
        interpolation_mat4 = cubic_spline_interpolation(pre_interpolation_feat4, 1)[:, 1:, :]
        interpolation_embed4 = self.traj_embed(interpolation_mat4)
        padding_embed5 = torch.zeros((batch_size, 32, self.gat_out_dim), device = x_dynamic.device, dtype=torch.float64)
        AR_embed5 = torch.cat([esa_output, pred_xy1_embed, pred_xy2_embed, pred_xy3_embed, pred_xy4_embed, \
                                interpolation_embed4, padding_embed5], dim = 1)
        AR_causal_mask5 = torch.zeros((batch_size, inp_his_len+63, inp_his_len+63), device = x_dynamic.device, dtype=torch.float64)
        AR_causal_mask5[:, inp_his_len+15:inp_his_len+31, :inp_his_len+31] = 1
        TrajAR_out5 = AR_embed5
        for trajAR_model in self.TrajAR:
            TrajAR_out5 = trajAR_model(TrajAR_out5, AR_causal_mask5)
        pred_xy5_embed = TrajAR_out5[:,inp_his_len+15:inp_his_len+31,:]
        out_traj_xy5 = self.out_linear2(TrajAR_out5)
        pred_xy5 = out_traj_xy5[:,inp_his_len+15:inp_his_len+31,:]
        # Sixth autoregression, generating all trajectory points
        pre_interpolation_feat5 = torch.cat([x_dynamic[:, 0, -1, :2].unsqueeze(1), pred_xy5], dim = 1)
        interpolation_mat5 = cubic_spline_interpolation(pre_interpolation_feat5, 1)[:, 1:, :]
        interpolation_embed5 = self.traj_embed(interpolation_mat5)
        AR_embed6 = torch.cat([esa_output, pred_xy1_embed, pred_xy2_embed, pred_xy3_embed, pred_xy4_embed, \
                                pred_xy5_embed, interpolation_embed5], dim = 1)
        AR_causal_mask6 = torch.zeros((batch_size, inp_his_len+63, inp_his_len+63), device = x_dynamic.device, dtype=torch.float64)
        AR_causal_mask6[:, inp_his_len+31:, :] = 1
        TrajAR_out6 = AR_embed6
        for trajAR_model in self.TrajAR:
            TrajAR_out6 = trajAR_model(TrajAR_out6, AR_causal_mask6)
        out_traj_xy6 = self.out_linear2(TrajAR_out6)
        pred_xy6 = out_traj_xy6[:,inp_his_len+31:,:]
        
        if is_training:
            # (batch,63,2)
            pred_traj = torch.cat([pred_xy1, pred_xy2, pred_xy3, pred_xy4, pred_xy5, pred_xy6], dim = 1)
            
            target_xy = torch.cat([y_targte[:, -1, :2].unsqueeze(1), y_targte[:, 15::16, :2], y_targte[:, 7::8, :2], \
                                   y_targte[:, 3::4, :2], y_targte[:, 1::2, :2], y_targte[:, :, :2]], dim = 1)
            if loss_func:
                xy_diff = pred_traj - target_xy
                dist = torch.mean(torch.mean(torch.sqrt(torch.sum(xy_diff ** 2, dim=2)), dim=1))
                return dist
        else:
            return pred_xy6
    

    def grad_state_dict(self):
        
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return  {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    
    def save(self, path:str):
        
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):

        loaded_params = torch.load(path)
        self.load_state_dict(loaded_params,strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params