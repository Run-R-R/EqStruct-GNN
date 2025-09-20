from typing import Union, Tuple
from torch import nn, Tensor
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool, Linear, GATConv, global_max_pool, GraphConv, GINConv
from torch_geometric.typing import Adj, OptTensor


class WeightedSAGEConv(SAGEConv):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'add', bias: bool = True):
        super().__init__(in_channels= in_channels, out_channels =out_channels, aggr=aggr)  # 选择聚合方式

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Adj,
        edge_weight: OptTensor = None
    ) -> Tensor:
        self._edge_weight = None
        out = super().forward(x, edge_index)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        if self._edge_weight is not None:
            return x_j * self._edge_weight.view(-1, 1)
        return x_j





class FairUserNet(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_channels, 2)
        self.fair_score_head = nn.Linear(hidden_channels + 2, 1)

    def forward(self, x_user, sensitive_attr=None):
        h = self.encoder(x_user)
        action_logits = self.action_head(h)
        fair_score = None
        if sensitive_attr is not None:
            s = F.one_hot(sensitive_attr, num_classes=2)
            s = s.view(-1, 2).float()
            h_fair = torch.cat([h, s], dim=-1)
            fair_score = self.fair_score_head(h_fair).squeeze()
        return action_logits, fair_score

class FairItemNet(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
        )
        # self.action_head = nn.Linear(hidden_channels, 2)
        self.fair_score_head = nn.Linear(hidden_channels + 1, 2)

    def forward(self, x_item, popularity=None):
        h = self.encoder(x_item)
        # action_logits = self.action_head(h)

        if popularity is not None:
            pop = popularity.view(-1, 1).float()
            pop = torch.sigmoid(pop)
            h_fair = torch.cat([h, pop], dim=-1)
            fair_score = self.fair_score_head(h_fair).squeeze()
        else:
            fair_score = None

        return fair_score


class EqStructGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,sensitive_attr, num_layers=2, tau=1.0,
                 num_users=6040, num_movies=3706):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, in_channels)
        self.movie_emb = nn.Embedding(num_movies, in_channels)
        self.user_proj = nn.Linear(in_channels, hidden_channels)
        self.item_proj = nn.Linear(in_channels, hidden_channels)
        self.num_users = num_users
        self.num_movies = num_movies
        self.in_channels = in_channels
        self.cls = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.ReLU()
        )
        self.x_dict = nn.ModuleDict({
            'user': nn.Embedding(self.num_users, self.in_channels),
            'item': nn.Embedding(self.num_movies, self.in_channels),
        })

        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.conv_first = HeteroConv({
            ('user', 'rates', 'item'): WeightedSAGEConv(in_channels, hidden_channels),
            ('item', 'rev_rates', 'user'): WeightedSAGEConv(in_channels, hidden_channels),

        }, aggr='mean')
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'rates', 'item'): WeightedSAGEConv(hidden_channels, hidden_channels),
                ('item', 'rev_rates', 'user'): WeightedSAGEConv(hidden_channels, hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)

            self.norms.append(nn.ModuleDict({
                'user': nn.LayerNorm(hidden_channels),
                'item': nn.LayerNorm(hidden_channels),
            }))
            self.dropouts.append(nn.Dropout(p=0.5))

        conv_end = HeteroConv({
            ('item', 'rev_rates', 'user'): WeightedSAGEConv( hidden_channels, out_channels),
            ('user', 'rates', 'item'): WeightedSAGEConv(hidden_channels, out_channels),
        }, aggr='max')
        self.convs.append(conv_end)
        self.user_fair_net = FairUserNet(hidden_channels, hidden_channels)
        self.item_fair_net = FairItemNet(hidden_channels, hidden_channels)

        self.sensitive_attr = sensitive_attr
        self.pool = global_mean_pool
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.tau = tau
    def fairness_loss_group(self, score: torch.Tensor, group: torch.Tensor):
        group0_score = score[group == 0]
        group1_score = score[group == 1]
        mean_gap = torch.abs(group0_score.mean() - group1_score.mean())
        return mean_gap


    def forward(self, data,sensitive_attr, item_pop, method="train"):
        x_dict, edge_index_dict, edge_label_dict = data.x_dict, data.edge_index_dict, data.edge_label_dict
        if method != "test":
            sensitive_attr = sensitive_attr[data["user"].n_id]
            item_pop = item_pop[data["item"].n_id]

        pos_edge_index_dict = {}
        neg_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type in edge_label_dict:
                pos_edge_mask = (edge_label_dict[edge_type] == 1).view(-1)
                neg_edge_mask = (edge_label_dict[edge_type] == 0).view(-1)

                pos_edge_index_filtered = edge_index[:, pos_edge_mask]
                neg_edge_index_filtered = edge_index[:, neg_edge_mask]
                pos_edge_index_dict[edge_type] = pos_edge_index_filtered
                neg_edge_index_dict[edge_type] = neg_edge_index_filtered
            else:
                pos_edge_index_dict[edge_type] = edge_index


        edge_index_dict = pos_edge_index_dict
        x_dict["user"] = self.user_proj(x_dict["user"])
        x_dict["item"] = self.item_proj(x_dict["item"])


        action_logits, fair_score = self.user_fair_net(x_dict['user'], sensitive_attr)  # shape: [B, 2], [B]
        item_logits = self.item_fair_net(x_dict['item'], item_pop)  # [num_items, 2]
        user_probs = F.gumbel_softmax(action_logits, tau=self.tau, hard=True)[:, 0]  #
        item_probs = F.gumbel_softmax(item_logits, tau=self.tau, hard=True)[:, 0]

        male_mask = (sensitive_attr == 0)
        female_mask = (sensitive_attr == 1)

        dp_male = user_probs[male_mask].sum().item()
        dp_female = user_probs[female_mask].sum().item()

        pos_edge_index = edge_index_dict[('user', 'rates', 'item')]  # shape [2, num_edges]

        user_pos_emb = x_dict['user'][pos_edge_index[0]]  # [num_pos_edges, dim]
        item_pos_emb = x_dict['item'][pos_edge_index[1]]  # [num_pos_edges, dim]
        pos_score = F.cosine_similarity(user_pos_emb, item_pos_emb)

        num_neg = pos_edge_index.size(1)
        num_users = x_dict['user'].size(0)
        num_items = x_dict['item'].size(0)
        user_neg_idx = torch.randint(0, num_users, (num_neg,), device=pos_edge_index.device)
        item_neg_idx = torch.randint(0, num_items, (num_neg,), device=pos_edge_index.device)

        user_neg_emb = x_dict['user'][user_neg_idx]
        item_neg_emb = x_dict['item'][item_neg_idx]
        neg_score = F.cosine_similarity(user_pos_emb, item_neg_emb) + F.cosine_similarity(user_neg_emb, item_pos_emb)

        temperature = 0.1
        pos_exp = torch.exp(pos_score / temperature)
        neg_exp = torch.exp(neg_score / temperature)
        struct_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()

        for i, conv in enumerate(self.convs):
            edge_weight_dict = {
                ('user', 'rates', 'item'): user_probs[edge_index_dict['user', 'rates', 'item'][0]] *
                                           item_probs[edge_index_dict['user', 'rates', 'item'][1]],
                ('item', 'rev_rates', 'user'): item_probs[edge_index_dict['item', 'rev_rates', 'user'][0]] *
                                               user_probs[edge_index_dict['item', 'rev_rates', 'user'][1]],
            }
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict)

            if i < len(self.norms):
                x_dict['user'] = self.norms[i]['user'](x_dict['user'])
                x_dict['item'] = self.norms[i]['item'](x_dict['item'])

                x_dict['user'] = self.dropouts[i](x_dict['user'])
                x_dict['item'] = self.dropouts[i](x_dict['item'])

        pooling = False
        if pooling:
            x_user = x_dict['user']  # [num_user, hidden_dim]
            x_item = x_dict['item']  # [num_item, hidden_dim]

        
            user2item_src = edge_index_dict[('user', 'rates', 'item')][0] 
            user2item_dst = edge_index_dict[('user', 'rates', 'item')][1]  

            item2user_src = edge_index_dict[('item', 'rev_rates', 'user')][0]  
            item2user_dst = edge_index_dict[('item', 'rev_rates', 'user')][1]  

            user_feat_on_edge = x_user[user2item_src]  # [E, hidden_dim]
            item_agg = self.pool(user_feat_on_edge, user2item_dst, size=x_item.size(0))  # [num_item, hidden_dim]

            item_feat_on_edge = x_item[item2user_src]
            user_agg = self.pool(item_feat_on_edge, item2user_dst, size=x_user.size(0))  # [num_user, hidden_dim]

            x_dict['user'] = user_agg + x_dict['user']
            x_dict['item'] = item_agg + x_dict['item']

        act = False
        if act:
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return x_dict, 1, struct_loss,user_probs
    
    def compute_bpr_loss(self, batch, x_dict):
        pos_edge_mask = (batch['user', 'rates', 'item'].edge_label == 1).view(-1)
        neg_edge_index_user = batch['user', 'rates', 'item'].edge_index[0][pos_edge_mask]
        neg_edge_index_movie = torch.randint(0, batch['item'].num_nodes, (len(neg_edge_index_user),),
                                             device=neg_edge_index_user.device)
        neg_edge_index = torch.stack([neg_edge_index_user, neg_edge_index_movie], dim=0)
        neg_pred = self.predict(x_dict['user'], x_dict['item'], neg_edge_index)

        pos_pred = self.predict(x_dict['user'], x_dict['item'], batch['user', 'rates', 'item'].edge_index[:, pos_edge_mask])

        diff = pos_pred - neg_pred  # [B]
        loss_pbr = -F.logsigmoid(diff).mean()
        return loss_pbr
        
    def predict(self, user_x, item_x, edge_index):
        user_emb = user_x[edge_index[0]]  # [num_edges, hidden]
        item_emb = item_x[edge_index[1]]  # [num_edges, hidden]
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores

    def link_pred(self, user_x, item_x, edge_index):
        user_emb = user_x[edge_index[0]]  # [num_edges, hidden]
        item_emb = item_x[edge_index[1]]  # [num_edges, hidden]
        concat = torch.cat([user_emb, item_emb], dim=-1)
        scores = self.cls(concat).squeeze(1)
        cls = F.sigmoid(scores)
        return cls

    def top_k_items_for_users(self, x_dict, users, k=5):
        user_emb = x_dict["user"][users]
        item_emb = x_dict["item"]
        socre_mat = user_emb @ item_emb.T
        values, indices = socre_mat.topk(k)
        return indices

