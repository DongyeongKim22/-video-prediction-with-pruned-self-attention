import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn.functional as F
from torch import nn, einsum
from preprocessing import CNNEncoder, PatchEmbedding, TemporalPositionalEncoding, VideoFrameEmbedding
from einops import rearrange, reduce, repeat
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
from torch.nn.functional import cosine_similarity
import time

from ITTR_pytorch import HPB
from ITTR_pytorch import DPSA


class ModifiedTransformerCNNForSegmentation(nn.Module):
    def __init__(self, num_classes=3, hpb_dims=[64, 128, 256], num_frames=4, d_model=256, nhead=8, num_encoder_layers=4):
        super(ModifiedTransformerCNNForSegmentation, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers   
        self.nhead = nhead   
        self.n_components = 1000
        self.temporal_pos_encoder = TemporalPositionalEncoding(d_model = self.n_components)
        
        encoder_layers = TransformerEncoderLayer(d_model=self.n_components, nhead=self.nhead).to(device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers).to(device)
        # CNN 인코더 초기화
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(num_classes, hpb_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[0]),
            nn.GELU(),
            nn.Conv2d(hpb_dims[0], hpb_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[0]),
            nn.GELU(),
            nn.Conv2d(hpb_dims[0], hpb_dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[1]),
            nn.GELU(),
            nn.Conv2d(hpb_dims[1], hpb_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[1]),
            nn.GELU(),
            nn.Conv2d(hpb_dims[1], hpb_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[2]),
            nn.GELU(),
            nn.Conv2d(hpb_dims[2], hpb_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[2]),
            nn.GELU())

        dim = hpb_dims[-1]
        self.hp_blocks = HPB(dim = hpb_dims[-1],         # dimension
                            dim_head = 28,     # dimension per attention head
                            heads = 8,         # number of attention heads
                            attn_height_top_k = int(28/2), # number of top indices to select along height, for the attention pruning
                            attn_width_top_k = int(28/2)  # number of top indices to select along width, for the attention pruning
                            ).to(device)
        # HPB 블록 초기화

        
        # 업스케일링을 위한 디코더
        self.seq_enc_reshaped = nn.Sequential(
                                nn.Linear(self.n_components, 256*28*28), # FC 레이어를 사용하여 차원 변환
                                nn.GELU())

        self.upscale = nn.Sequential(
            nn.Conv2d(hpb_dims[2]*self.num_frames, hpb_dims[2]*self.num_frames, kernel_size=3, 
                               padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hpb_dims[2]*self.num_frames, int(hpb_dims[2]*self.num_frames/2), kernel_size=3, 
                               padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(int(hpb_dims[2]*self.num_frames/2), int(hpb_dims[2]*self.num_frames/2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(hpb_dims[2]*self.num_frames/2)),
            nn.GELU(),
            nn.Conv2d(int(hpb_dims[2]*self.num_frames/2), hpb_dims[2], kernel_size=3, 
                               padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hpb_dims[2], hpb_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[2]),
            nn.GELU(),
            nn.ConvTranspose2d(hpb_dims[2], hpb_dims[1], kernel_size=3, stride=2, 
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[1]),
            nn.GELU(),
            nn.Conv2d(hpb_dims[1], hpb_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[1]),
            nn.GELU(),
            
            nn.ConvTranspose2d(hpb_dims[1], hpb_dims[0], kernel_size=3, stride=2, 
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[0]),
            nn.GELU(),
            nn.Conv2d(hpb_dims[0], hpb_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hpb_dims[0]),
            nn.GELU(),
            nn.Conv2d(hpb_dims[0], hpb_dims[0]*2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hpb_dims[0]*2),
            nn.ConvTranspose2d(hpb_dims[0]*2, num_classes*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(num_classes*2),
            nn.Conv2d(num_classes*2, num_classes, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
          )
        
    def forward(self, video):
        batch_size, n_frames, C, H, W = video.shape  # video shape: (batch_size, num_frames, channels, H, W)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 각 프레임을 독립적으로 처리
        x = video.view(batch_size * n_frames, C, H, W)

        x = self.cnn_encoder(x)
        
        start_time = time.time()
        
        x = self.hp_blocks(x)
        
        x_temporal = x.view(batch_size * self.num_frames, -1)
        
        # 랜덤 투영 가중치 생성
        n_samples, n_features = x_temporal.shape
        projection_matrix = torch.randn(n_features, self.n_components) / self.n_components

        # 랜덤 투영 적용
        x_temporal = torch.mm(x_temporal, projection_matrix.to(device))
        x_temporal = x_temporal.view(batch_size, self.num_frames, -1)

        # similarity = cosine_similarity(x_temporal[i].unsqueeze(0), x.view(batch_size * self.num_frames, -1)[j].unsqueeze(0))
        
        x_temporal = self.temporal_pos_encoder(x_temporal)
        
        # 트랜스포머 인코더 적용
        x_temporal = x_temporal.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
        x_temporal = self.transformer_encoder(x_temporal).view(batch_size*self.num_frames, -1)
        
        x = x.view(batch_size, self.num_frames, x.size(1), x.size(2), x.size(3))
        #add residual
        
        x_temporal = self.seq_enc_reshaped(x_temporal)
        x_temporal = x_temporal.view(x.size())
        x = (x + x_temporal)
        x = x.view(batch_size, n_frames*x.size(2), x.size(3), x.size(4))
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Execution time baseline: {execution_time:.4f} seconds")
        x = self.upscale(x)

        return x




