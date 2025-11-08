import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, layer_norm=False, batch_norm=False):
        super(Encoder, self).__init__()

        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.mlp = torch.nn.Linear(in_channels, hidden_channels)
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.bn = torch.nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        x = self.mlp(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.layer_norm:
            x = self.ln(x)
        if self.batch_norm:
            x = self.bn(x)
        return x


class GCL_CS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, alpha, temperature, dropout, layer_norm=False, batch_norm=False):
        super(GCL_CS, self).__init__()

        self.alpha = alpha
        self.temperature = temperature
        self._combination = False

        self.enc = Encoder(in_channels, hidden_channels, dropout, layer_norm, batch_norm)
        self.n_linear = torch.nn.Linear(hidden_channels, hidden_channels)
        self.p_linear = torch.nn.Linear(hidden_channels, hidden_channels)

    def dual_combination(self, x, node_to_par, P, A_P):
        ###  node-level
        q_n = F.sigmoid(x)
        k_n = torch.spmm(P.T, q_n)
        s_n = torch.sum(q_n, dim=0)
        pos_n = torch.sum(q_n * k_n[node_to_par], dim=1)
        neg_n = torch.sum(q_n * s_n, dim=1)

        ### community-level
        k_p = F.normalize(torch.spmm(P.T, x))
        neg_p = torch.exp(torch.mm(k_p, k_p.t()) / self.temperature)
        pos_p = torch.sum(A_P * neg_p, dim=1)
        neg_p = torch.sum(neg_p, dim=1)

        loss = - torch.log(self.alpha * pos_n + (1 - self.alpha) * pos_p[node_to_par]) + \
               torch.log(self.alpha * neg_n + (1 - self.alpha) * neg_p[node_to_par])
        return loss.mean()

    def dual_product(self, x, node_to_par, P, A_P):
        q_n = F.sigmoid(x)
        k_p = F.normalize(torch.spmm(P.T, x))
        attention_p = torch.exp(torch.mm(k_p, k_p.t()) / self.temperature)
        k_n = torch.spmm(P.T, q_n)

        pos_message = torch.mm(attention_p*A_P, k_n)
        pos_score = torch.sum(q_n * pos_message[node_to_par], dim=1)

        neg_message = torch.mm(attention_p, k_n)
        neg_score = torch.sum(q_n * neg_message[node_to_par], dim=1)

        loss = -torch.log(pos_score) + torch.log(neg_score)
        return loss.mean()

    def forward(self, x, node_to_par, P, A_P):
        x = self.enc(x)
        if self._combination:
            loss = self.dual_combination(x, node_to_par, P, A_P)
        else:
            loss = self.dual_product(x, node_to_par, P, A_P)
        return loss

    def get_embed(self, x):
        x = self.enc(x)
        return x
