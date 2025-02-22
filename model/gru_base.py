import torch.nn as nn
import torch


class GRUModel(nn.Module):
    def __init__(
            self,
            input_num: int = 6,
            hidden_num: int = 30,
            head_type: str = 'prediction',
            head_dropout: int = 0.0):
        super().__init__()

        assert head_type in ['pretrain', 'prediction'], 'head type should be either pretrain or prediction'

        self.gru = nn.GRU(input_num, hidden_num, batch_first=True, num_layers=1)
        if head_type == "pretrain":
            self.head = PretrainHead(hidden_num, input_num, head_dropout)
        elif head_type == "prediction":
            self.head = PredictionHead(hidden_num, head_dropout)

    def forward(self, x1, mask_point=None):  # (bat_size, seq, features)
        l_x1, _ = self.gru(x1) #  5000, 40, 30
        output = self.head(l_x1, mask_point)
        return output, _


class PretrainHead(nn.Module):
    def __init__(self, hidden_num, feature_num, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_num, feature_num)

    def forward(self, x):
        """
        x: tensor [bs x seq_len x hidden_num]
        output: tensor [bs x seq_len x feature_num]
        """

        x = self.linear(self.dropout(x))
        return x


class PredictionHead(nn.Module):
    def __init__(self, hidden_num, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.BatchNorm1d(hidden_num)
        self.linear = nn.Linear(hidden_num, 1)

    def forward(self, x, mask_point=None):
        """
        x: tensor [bs x seq_len x hidden_num]
        mask_point: [bs]
        output: tensor [bs]
        """
        x = self.dropout(x)
        if mask_point is None:
            output = self.linear(self.hidden(x[:, -1, :]))
        else:
            bs = x.shape[0]
            output = self.linear(self.hidden(x[torch.arange(0, bs).long(), mask_point.long(), :]))
        return output


