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

from ITTR_pytorch import HPB
from ITTR_pytorch import DPSA


class Transformer_UNet(nn.Module):
    def __init__(self, num_classes=3, hpb_dims=[64, 128, 256], num_frames=4, d_model=256, nhead=8, num_encoder_layers=4):
        super(Transformer_UNet, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers   
        self.nhead = nhead   
        self.n_components = 1000
        self.temporal_pos_encoder = TemporalPositionalEncoding(d_model = self.n_components)
        
        encoder_layers = TransformerEncoderLayer(d_model=self.n_components, nhead=self.nhead).to(device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers).to(device)
        # CNN encoder
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

        self.seq_enc_reshaped = nn.Sequential(
                                nn.Linear(self.n_components, 256*28*28),
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


        x = video.view(batch_size * n_frames, C, H, W)
        
        cnn_features = []
        for i in range(len(self.cnn_encoder)):
            x = self.cnn_encoder[i](x)
            if isinstance(self.cnn_encoder[i], nn.Conv2d) and i == 15 :
                last_frame_indices = [(j+1) * n_frames - 1 for j in range(batch_size)]
                last_frames_features = x[last_frame_indices, :, :, :]
                cnn_features.append(last_frames_features)

        
        
        x = self.hp_blocks(x)
        
        x_temporal = x.view(batch_size * self.num_frames, -1)
        

        n_samples, n_features = x_temporal.shape
        projection_matrix = torch.randn(n_features, self.n_components) / self.n_components


        x_temporal = torch.mm(x_temporal, projection_matrix.to(device))
        x_temporal = x_temporal.view(batch_size, self.num_frames, -1)

        # similarity = cosine_similarity(x_temporal[i].unsqueeze(0), x.view(batch_size * self.num_frames, -1)[j].unsqueeze(0))
        
        x_temporal = self.temporal_pos_encoder(x_temporal)

        x_temporal = x_temporal.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
        x_temporal = self.transformer_encoder(x_temporal).view(batch_size*self.num_frames, -1)
        
        x = x.view(batch_size, self.num_frames, x.size(1), x.size(2), x.size(3))
        #add residual
        
        x_temporal = self.seq_enc_reshaped(x_temporal)
        x_temporal = x_temporal.view(x.size())
        x = (x + x_temporal)
        x = x.view(batch_size, -1, x.size(3), x.size(4))
        
        for i, layer in enumerate(self.upscale):
            if (isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)) and i==9:
                skip_feature = cnn_features.pop()
                dim = x.size(1)
                x = torch.cat([x, skip_feature], dim = 1)
                dynamic_conv = nn.Conv2d(in_channels=x.size(1), out_channels=dim, kernel_size=3, padding=1, bias=False).to(x.device)
                x = dynamic_conv(x)
                x = F.gelu(x)
            x = layer(x)


        return x




