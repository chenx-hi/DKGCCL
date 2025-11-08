import torch
import torch.nn.functional as F


def get_PGD_inputs(gcl_encoder, x, center, cluster_y, criterion, eps):
    gcl_encoder.train()
    iters = 2 #3
    alpha = eps / 4
    gcl_encoder.dropout = 0

    # init
    delta = torch.rand(x.shape) * eps * 2 - eps
    delta = delta.to(x.device)
    delta = torch.nn.Parameter(delta)

    for i in range(iters):
        p_x = x + delta
        p_x = gcl_encoder(p_x)
        logits = torch.matmul(p_x, center.T)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, cluster_y)
        loss.backward()

        # delta update
        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)

    output = delta.detach()
    return gcl_encoder(output+x)


class Distiller(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Distiller, self).__init__()

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x
