import torch.nn as nn
import torch
import math


class GRUPatchModel(nn.Module):
    def __init__(
            self,
            input_dim=6,
            hidden_dim=30,
            N_patch=5,
            step=1,
            seq_len=40,
            patch_axis=0):
        super().__init__()
        assert patch_axis in [0, 1], 'patch should be either 0 or 1'
        self.N_patch = N_patch
        self.step = step
        self.patch_num = math.floor((seq_len - N_patch)/step) + 1
        self.input_dim = input_dim
        self.patch_axis = patch_axis
        self.gru = nn.GRU(N_patch, hidden_dim, batch_first=True, num_layers=1)
        # self.flatten = Flatten_Head(num_feature=input_dim, nf=hidden_dim * patch_num, last_day=False)
        self.flatten = Flatten_Head(num_feature=input_dim, nf=hidden_dim)

    def forward(self, x1):  # 5000, 40, 6
        x1 = x1.permute(0, 2, 1).contiguous()  # 5000, 6, 40

        # patch
        x1 = x1.unfold(dimension=-1, size=self.N_patch, step=self.step)  # 5000, 6, 8, 5
        if self.patch_axis == 1:
            x1 = x1.permute(0, 1, 3, 2).contiguous()

        # CI
        x1 = torch.reshape(x1, (x1.shape[0] * x1.shape[1], x1.shape[2], x1.shape[3]))  # 5000*6, 8, 5

        # gru
        x1, _ = self.gru(x1)  # 5000*6, 8, 30

        # uCI
        x1 = torch.reshape(x1, (-1, self.input_dim, x1.shape[1], x1.shape[2]))  # 5000, 6, 8, 30

        # Flatten
        return self.flatten(x1), 0


class Flatten_Head(nn.Module):
    def __init__(self, num_feature: int, nf: int, head_dropout: int = 0.1, last_day: bool = True):
        super().__init__()
        self.num_feature = num_feature
        self.last_day = last_day

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()

        for i in range(self.num_feature):
            self.flattens.append(nn.Flatten(start_dim=-2))
            self.linears.append(nn.Linear(nf, 1))
            self.dropouts.append(nn.Dropout(head_dropout))

        self.batch0 = nn.BatchNorm1d(nf)
        self.batch1 = nn.BatchNorm1d(num_feature)

        self.outlinear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_feature, 1))

    def forward(self, x):
        x_out = []
        for i in range(self.num_feature):
            if self.last_day:
                z = self.linears[i](self.dropouts[i](x[:, i, -1, :]))  # z: 5000, 1
            else:
                z = self.flattens[i](x[:, i, :, :])  # z: 5000, 8*30
                z = self.linears[i](self.dropouts[i](z))  # z: 5000, 1
            x_out.append(z)
        x = torch.stack(x_out, dim=1)  # x: 5000, 6, 1
        x = torch.squeeze(x, -1)  # 5000, 6
        return self.outlinear(self.batch1(x))  # 5000, 1
