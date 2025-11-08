import torch

from modules.distiller import Distiller
from modules.gcl import GCL_CS


class DualGCL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, alpha, temperature, dropout, layer_norm=False, batch_norm=False):
        super(DualGCL, self).__init__()

        self._kd = False

        self.gcl_model = GCL_CS(in_channels, hidden_channels, alpha, temperature, dropout, layer_norm, batch_norm)
        self.kd_model = Distiller(hidden_channels)
        self.kd_test = Distiller(hidden_channels)

    def weights_init(self):
        self.gcl_model.enc.ln.reset_parameters()
        self.gcl_model.enc.bn.reset_parameters()
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def gcl_embed(self, x):
        x = self.gcl_model.get_embed(x)
        return x

    #products
    def adj_embed(self, x, g, k):
        x = self.gcl_model.get_embed(x)
        for _ in range(k):
            x = torch.spmm(g, x)
        return torch.nn.functional.relu(x)

    def adj_embed_add(self, x, g, k):
        x = self.gcl_model.get_embed(x)
        if k==0:
            output =x
        else:
            output = 0.05*x
        for _ in range(k):
            x = torch.spmm(g, x)
            output = output + x/k
        return torch.nn.functional.relu(output)

    def kd_embed(self, x):
        x = self.kd_model(x)
        return x






