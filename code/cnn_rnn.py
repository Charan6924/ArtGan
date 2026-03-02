import torch
import torch.nn as nn
import torchvision.models as models


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_channels = hidden_channels

        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList() if bidirectional else None
        self.dropout_layers = nn.ModuleList()

        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels * (2 if bidirectional else 1)
            self.forward_cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
            if bidirectional:
                self.backward_cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
            if i < num_layers - 1:
                self.dropout_layers.append(nn.Dropout2d(dropout))

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.shape
        device = x.device

        for layer_idx in range(self.num_layers):
            forward_cell = self.forward_cells[layer_idx]
            h, c = forward_cell.init_hidden(batch_size, height, width, device)
            forward_outputs = []
            for t in range(seq_len):
                h, c = forward_cell(x[:, t], (h, c))
                forward_outputs.append(h)

            if self.bidirectional and self.backward_cells is not None:
                backward_cell = self.backward_cells[layer_idx]
                h, c = backward_cell.init_hidden(batch_size, height, width, device)
                backward_outputs = []
                for t in range(seq_len - 1, -1, -1):
                    h, c = backward_cell(x[:, t], (h, c))
                    backward_outputs.insert(0, h)
                outputs = [torch.cat([forward_outputs[t], backward_outputs[t]], dim=1) for t in range(seq_len)]
            else:
                outputs = forward_outputs

            x = torch.stack(outputs, dim=1)

            if layer_idx < self.num_layers - 1:
                x = x.view(batch_size * seq_len, -1, height, width)
                x = self.dropout_layers[layer_idx](x)
                x = x.view(batch_size, seq_len, -1, height, width)

        return x

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1)
        )

    def forward(self, x):
        seq_len = x.shape[1]
        scores = []
        for i in range(seq_len):
            scores.append(self.attention(x[:, i]))
        scores = torch.softmax(torch.cat(scores, dim=1), dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weighted = (x * scores).sum(dim=1)
        return weighted


class ArtClassifier(nn.Module):
    def __init__(
        self,
        num_styles=27,
        num_artists=1119,
        rnn_hidden=256,
        rnn_layers=2,
        dropout=0.3,
        freeze_backbone=True
    ):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.feature_dim = 2048
        self.spatial_size = 7

        self.channel_reduce = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.convlstm = ConvLSTM(
            input_channels=512,
            hidden_channels=rnn_hidden,
            kernel_size=3,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout
        )

        rnn_output_channels = rnn_hidden * 2

        self.temporal_attention = TemporalAttention(rnn_output_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        self.style_head = nn.Sequential(
            nn.Linear(rnn_output_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_styles)
        )

        self.artist_head = nn.Sequential(
            nn.Linear(rnn_output_channels, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_artists)
        )

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x, task='all'):
        features = self.extract_features(x)
        features = self.channel_reduce(features)

        b, c, h, w = features.shape
        seq = features.permute(0, 2, 1, 3).reshape(b, h, c, 1, w)

        rnn_out = self.convlstm(seq)
        pooled = self.temporal_attention(rnn_out)
        pooled = self.global_pool(pooled).flatten(1)
        pooled = self.dropout(pooled)

        if task == 'style':
            return self.style_head(pooled)
        elif task == 'artist':
            return self.artist_head(pooled)
        else:
            return {
                'style': self.style_head(pooled),
                'artist': self.artist_head(pooled)
            }

    def get_embedding(self, x):
        features = self.extract_features(x)
        features = self.channel_reduce(features)
        b, c, h, w = features.shape
        seq = features.permute(0, 2, 1, 3).reshape(b, h, c, 1, w)
        rnn_out = self.convlstm(seq)
        pooled = self.temporal_attention(rnn_out)
        return self.global_pool(pooled).flatten(1)

    def unfreeze_backbone(self, layers=-1):
        if layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            children = list(self.backbone.children())
            for child in children[layers:]:
                for param in child.parameters():
                    param.requires_grad = True


def build_model(config=None):
    defaults = {
        'num_styles': 27,
        'num_artists': 1119,
        'rnn_hidden': 256,
        'rnn_layers': 2,
        'dropout': 0.4,
        'freeze_backbone': True
    }
    if config:
        defaults.update(config)
    return ArtClassifier(**defaults)
