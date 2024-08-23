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

'''
# Example usage
model = UNetRes(n_channels=3, n_classes=3, bilinear=True, time_embed_dim=128)
input_image = torch.randn(2, 3, 256, 256)  # Batch size 1, RGB image, 256x256
time_step = torch.tensor([10.0, 9.0]).reshape(2,1)  # Example time step
output_image = model(input_image, time_step)
print(output_image.shape)  # Should be (1, 3, 256, 256)
'''


import numpy as np
import matplotlib.pyplot as plt
seq_dim=(8,8)

def fm_loss_CNN2d(model, samples, t, u_t_label, epoch, plot_u=False):
    samples.requires_grad_(True)
    batch_size = samples.shape[0]
    u_t_model = model(samples, t).reshape(batch_size,np.prod(seq_dim))
    # Compute the norm loss
    if plot_u:
        plt.figure(figsize=(11,5))
        plt.subplot(121)
        plt.scatter(samples.reshape(batch_size,np.prod(seq_dim))[:,0].cpu().detach().numpy(),u_t_label[:,0].cpu().detach().numpy(),s=3)
        plt.ylabel("u_t_label")
        plt.xlabel("x_t")
        plt.subplot(122)
        plt.scatter(samples.reshape(batch_size,np.prod(seq_dim))[:,0].cpu().detach().numpy(),u_t_model[:,0].cpu().detach().numpy(),s=3)
        plt.ylabel("u_t_model")
        plt.xlabel("x_t")
        plt.savefig("prediction_u-epoch%d"%epoch, bbox_inches="tight")
        plt.show()
    norm_loss = torch.norm(u_t_label - u_t_model, dim = -1) ** 2 / 2.
    return norm_loss

import torch.optim as optim

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}

        # Initialize the shadow model with the original model's parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        # Update the EMA model with the latest model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # Replace model parameters with EMA parameters for evaluation
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()

    def restore(self):
        # Restore the model's original parameters after evaluation
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()


# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using cuda:: ", torch.cuda.is_available())

model_u_t = UNetRes(1,1,bilinear=True, time_embed_dim=128)
model_u_t = model_u_t.to(device)
# ema = EMA(model_u_t, decay=0.999)
optimizer = optim.Adam(model_u_t.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


xx_t = torch.from_numpy(np.load("data_xx_t.npy")).float().to(device)
u_t_xx = torch.from_numpy(np.load("data_u_t.npy")).float().to(device)
beta_t = torch.from_numpy(np.load("data_beta_tt.npy")).float().to(device)
NT = u_t_xx.shape[0]
NS = u_t_xx.shape[1]
eval_xx_t = torch.from_numpy(np.load("eval_xx_t.npy")).float().to(device)
eval_u_t_xx = torch.from_numpy(np.load("eval_u_t.npy")).float().to(device)
eval_beta_t = torch.from_numpy(np.load("eval_beta_tt.npy")).float().to(device)
NT_eval = eval_u_t_xx.shape[0]
NS_eval = eval_u_t_xx.shape[1]
l_curve = []
batch_size = 512
samples = np.zeros((batch_size, 8, 8))
print('[Time step, loss value, lr]')
# sigma_t = np.array([np.sqrt(2./beta_tt) for beta_tt in beta_t])
for t in range(100000):
    model_u_t.train()
    # idx = np.random.choice(NS * NT, batch_size)
    idx = torch.randint(0, NS * NT, (batch_size,))
    t_idx = idx // NS
    x_idx = idx % NS
    # samples[:,0] = beta_t[t_idx]
    # samples[:,1:] = xx_t[t_idx, x_idx, :]
    samples = xx_t[t_idx, x_idx, :].reshape(batch_size,1,8,8)
    u_t_label = u_t_xx[t_idx, x_idx, :].reshape(batch_size,64)
    samples_beta_t = beta_t[t_idx, x_idx].reshape(batch_size, 1)
    loss = fm_loss_CNN2d(model_u_t, samples, samples_beta_t, u_t_label, t)
    loss = (loss*samples_beta_t.reshape(-1)).mean(-1)
    # Before the backward pass, zero all of the network gradients
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to parameters
    loss.backward()
    # Calling the step function to update the parameters
    optimizer.step()
    # scheduler.step(loss)
    # ema.update()
    if ((t % 10) == 0):
        l_curve.append([t,loss.item()])
    if ((t % 10) == 0):
        print([t,loss.item()])
    if ((t % 500) == 0):
        # ema.apply_shadow()
        idx = torch.randint(0, NS_eval * NT_eval, (512,))
        t_idx = idx // NS_eval
        x_idx = idx % NS_eval
        model_u_t.eval()
        loss_eval = fm_loss_CNN2d(model_u_t, eval_xx_t[t_idx, x_idx].reshape(512,1,8,8), eval_beta_t[t_idx, x_idx].reshape(512,1), eval_u_t_xx[t_idx, x_idx].reshape(512,64), t, plot_u=True)
        loss_eval = (loss_eval*eval_beta_t[t_idx, x_idx].reshape(-1)).mean(-1)
        print([t,loss_eval.item()])
        if loss_eval > 10*loss:
            raise Exception("Overfitting:: loss_eval = %f    loss_train = %f"%(loss_eval.item(), loss.item()))
    if ((t % 2000) == 0):
        torch.save(model_u_t.state_dict(), f'ckpt_unetvelocity.pth')
        torch.save(optimizer.state_dict(), f"ckpt_unetvelocity_optimizer.pth")
        torch.save(model_u_t.state_dict(), f'ckpt_unetvelocity-epoch{t}.pth')
        torch.save(optimizer.state_dict(), f"ckpt_unetvelocity_optimizer-epoch{t}.pth")