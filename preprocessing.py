import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256]):
        super(CNNEncoder, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = hidden_dim
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        H_new, W_new = x.shape[2], x.shape[3]
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                H_new, W_new = H_new // layer.stride, W_new // layer.stride
        return x, H_new, W_new

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=100, input_channels=256):  # input_channels 파라미터 추가
        super().__init__()
        self.emb_size = emb_size
        self.projection = nn.Linear(input_channels, emb_size)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        x = self.projection(x)
        return x

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(TemporalPositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        pe = self.pe[:x.size(1)]
        pe = pe.unsqueeze(0).expand(x.size(0), -1, -1)
        x = x + pe.to(device)
        return x

class VideoFrameEmbedding(nn.Module):
    def __init__(self, n_frames=4, d_model=100):
        super(VideoFrameEmbedding, self).__init__()
        self.n_frames = n_frames
        # 인스턴스화를 확인하고 조건부로 실행
        self.cnn_encoder = CNNEncoder()
        self.patch_embedding = PatchEmbedding(emb_size=d_model, input_channels=256)  
        # `input_channels`의 값을 CNN 출력에 맞추어 설정
        self.temporal_pos_encoding = TemporalPositionalEncoding(d_model)

    def forward(self, video):
        batch_size, n_frames, C, H, W = video.size()
        video = video.view(batch_size * n_frames, C, H, W)
        frame_features, H_new, W_new = self.cnn_encoder(video)
        patch_embeddings = self.patch_embedding(frame_features)
        video_embedding = patch_embeddings.view(batch_size, n_frames*H_new*W_new, -1)
        video_embedding = self.temporal_pos_encoding(video_embedding)
        return video_embedding





