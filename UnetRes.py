import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """A simple Residual Block with time embedding"""
    def __init__(self, in_channels, out_channels, time_embed_dim=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else None
        
        self.time_embed_layer = nn.Sequential(
            nn.Linear(time_embed_dim, out_channels),
            nn.ReLU()
        ) if time_embed_dim is not None else None

    def forward(self, x, t=None):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        
        if self.time_embed_layer is not None and t is not None:
            t = self.time_embed_layer(t).unsqueeze(-1).unsqueeze(-1)
            out = out + t
        
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip_conv:
            identity = self.skip_conv(identity)
            
        out += identity
        out = self.relu(out)
        
        return out

class Down(nn.Module):
    """Downscaling with maxpool then resblock"""
    def __init__(self, in_channels, out_channels, time_embed_dim=None):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool2d(1)
        self.resblock = ResBlock(in_channels, out_channels, time_embed_dim)

    def forward(self, x, t=None):
        x = self.maxpool(x)
        x = self.resblock(x, t)
        return x

class Up(nn.Module):
    """Upscaling then resblock"""
    def __init__(self, in_channels, out_channels, bilinear=True, time_embed_dim=None):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResBlock(in_channels, out_channels, time_embed_dim)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResBlock(in_channels, out_channels, time_embed_dim)

    def forward(self, x1, x2, t=None):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class TimeEmbedding(nn.Module):
    """Simple Time Embedding Layer"""
    def __init__(self, time_dim, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t):
        return self.time_embed(t)

class UNetRes(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, time_embed_dim=128):
        super(UNetRes, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.time_embedding = TimeEmbedding(time_dim=1, embed_dim=time_embed_dim)

        self.inc = ResBlock(n_channels, 64, time_embed_dim)
        self.down1 = Down(64, 128, time_embed_dim)
        self.down2 = Down(128, 256, time_embed_dim)
        self.down3 = Down(256, 512, time_embed_dim)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, time_embed_dim)
        self.up1 = Up(1024, 512 // factor, bilinear, time_embed_dim)
        self.up2 = Up(512, 256 // factor, bilinear, time_embed_dim)
        self.up3 = Up(256, 128 // factor, bilinear, time_embed_dim)
        self.up4 = Up(128, 64, bilinear, time_embed_dim)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        
        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)
        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        logits = self.outc(x)
        return logits