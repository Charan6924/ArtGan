import torch
import torch.nn as nn
import torchvision.models as models


class SpatialFeatureSequence(nn.Module):
    def __init__(self, method='row'):
        super().__init__()
        self.method = method

    def forward(self, x):
        b, c, h, w = x.shape
        if self.method == 'row':
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        elif self.method == 'column':
            x = x.permute(0, 3, 2, 1).reshape(b, h * w, c)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return (x * weights).sum(dim=1)


class ArtClassifier(nn.Module):
    def __init__(
        self,
        num_styles=27,
        num_artists=1119,
        rnn_hidden=512,
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
        self.seq_len = 49  

        self.to_sequence = SpatialFeatureSequence(method='row')

        self.rnn = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

        rnn_output_size = rnn_hidden * 2  # bidirectional

        self.attention = Attention(rnn_output_size)
        self.dropout = nn.Dropout(dropout)

        self.style_head = nn.Sequential(
            nn.Linear(rnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_styles)
        )

        self.artist_head = nn.Sequential(
            nn.Linear(rnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_artists)
        )

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x, task='all'):
        features = self.extract_features(x)
        seq = self.to_sequence(features)

        rnn_out, _ = self.rnn(seq)
        pooled = self.attention(rnn_out)
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
        seq = self.to_sequence(features)
        rnn_out, _ = self.rnn(seq)
        return self.attention(rnn_out)

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
        'rnn_hidden': 512,
        'rnn_layers': 2,
        'dropout': 0.4,
        'freeze_backbone': True
    }
    if config:
        defaults.update(config)
    return ArtClassifier(**defaults)
