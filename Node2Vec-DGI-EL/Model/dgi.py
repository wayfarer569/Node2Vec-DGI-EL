import dgl
import torch
import logging
import dgl.nn as dglnn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np


logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, dim=0, keepdim=True)  # 输出形状 (1, hidden_dim)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)  # 仍然保持 (1, hidden_dim)

class GATFeatureAggregator(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, dropout=0.2):
        super(GATFeatureAggregator, self).__init__()

        self.layer1 = dglnn.GATConv(in_feats, hidden_feats, num_heads)
        self.layer2 = dglnn.GATConv(hidden_feats * num_heads, out_feats,1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, g, inputs, sparse):
        if sparse:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
        h = self.layer1(g, inputs)
        h = self.dropout(F.elu(h).view(h.shape[0], -1))
        h = self.layer2(g, h)
        return h.squeeze(1)

class DGI(nn.Module):
    def __init__(self, gat_model, hidden_dim):
        super(DGI, self).__init__()
        self.gat_model = gat_model
        self.scoring_function = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg_readout = AvgReadout()


    def forward(self, graph, pos_features, neg_features, sparse=True, pos_bias=None, neg_bias=None, msk=None):

        pos_embeddings = self.gat_model(graph, pos_features, sparse)
        global_positive = self.avg_readout(pos_embeddings, msk)
        global_positive = self.sigmoid(global_positive)
        neg_embeddings = self.gat_model(graph, neg_features, sparse)

        expanded_global_positive = global_positive.expand(pos_embeddings.size(0), -1)
        expanded_global_negative = global_positive.expand(neg_embeddings.size(0), -1)

        pos_scores = torch.squeeze(self.scoring_function(pos_embeddings, expanded_global_positive), 1)
        neg_scores = torch.squeeze(self.scoring_function(neg_embeddings, expanded_global_negative), 1)

        if pos_bias is not None:
            pos_scores += pos_bias
        if neg_bias is not None:
            neg_scores += neg_bias

        logits = torch.cat((pos_scores, neg_scores), dim=0)
        return logits



def train_dgi(dgi_model, graph, features, num_epochs, lr, file2, weight_decay=1e-5, sparse=False):
    b_xent = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(dgi_model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float('inf')
    cnt_wait = 0
    for epoch in range(num_epochs):
        dgi_model.train()
        optimizer.zero_grad()

        idx = np.random.permutation(features.size(0))  # 假设features的第一个维度是节点数量
        shuffled_features = features[idx]  # 只需打乱第一个维度

        labels_pos = torch.ones(features.size(0))  # 正样本标签   #    lbl_1 = torch.ones(batch_size, nb_nodes)
        labels_neg = torch.zeros(features.size(0))  # 负样本标签  #    lbl_2 = torch.zeros(batch_size, nb_nodes)
        labels = torch.cat((labels_pos, labels_neg), dim=0)  # 合并标签

        logits = dgi_model(graph, features, shuffled_features, sparse=sparse)

        if labels.device != logits.device:
            labels = labels.to(logits.device)
        loss = b_xent(logits, labels)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        if loss < best_loss:
            best_loss = loss
            torch.save(dgi_model.state_dict(), file2)   # 检查并保存最佳模型
            cnt_wait = 0
        else:
            cnt_wait += 1
        if cnt_wait > 20:
            print("Early stopping...")
            break


def DGI_GAT_main(train_graph,DGI_emb_save,file1,file2,hid_emb,out_emb,num_heads,num_epochs ,lr):
    features = train_graph.ndata['feat']
    DGI_in_feats = features.shape[1]
    DGI_hidden_feats1 = hid_emb
    DGI_out_feats = out_emb
    DGI_num_heads = num_heads
    DGI_num_epochs =num_epochs
    DIG_lr = lr
    aggregator_model = GATFeatureAggregator(DGI_in_feats, DGI_hidden_feats1, DGI_out_feats, DGI_num_heads)
    # 初始化 DGI 模型
    dgi_model = DGI(aggregator_model, DGI_out_feats)
    dgi_model.to(device)
    features = features.to(device)
    train_graph = train_graph.to(device)

    train_dgi(dgi_model, train_graph, features, DGI_num_epochs, DIG_lr,file2)
    dgi_model.load_state_dict(torch.load(file2))
    dgi_model.eval()
    with torch.no_grad():
        # 使用 GAT 模型来获取嵌入向量
        embeddings_np = dgi_model.gat_model(train_graph, features,True)
        embeddings = embeddings_np.cpu().numpy()
        if DGI_emb_save is True:
            np.save(file1, embeddings)
    return embeddings_np
