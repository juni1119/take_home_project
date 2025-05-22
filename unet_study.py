import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Positional Embedding ---
class SinusoidalPositionEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
      #  4) dim 만큼 t 값을 확장한다 (64,) -> (64, dim)
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).to(t.device)
        emb = t[:, None] * emb[None, :]
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)

# --- UNet ---
class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmb(time_emb_dim),
            # 3) SinusoidalPositionEmb forward 실행 
            nn.Linear(time_emb_dim, time_emb_dim),
            # 학습될 network
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.down = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv4 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.out = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        # 2) time step t를 self.time_mlp에 입력

        x1 = F.relu(self.conv1(x) + t_emb)
        x2 = F.relu(self.conv2(x1))
        x3 = self.down(x2)
        x4 = F.relu(self.conv3(x3) + t_emb)

        x5 = self.up(x4)
        x6 = F.relu(self.conv4(x5 + x2))  # skip connection
        # conv1, 2, 3, 4 : 학습될 network 

        out = self.out(x6)
        return out  # predicted ε

# --- Training Function ---
def train_unet(model, dataloader, optimizer, device, T=1000, num_epochs=10):
    model.train()
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    for epoch in range(num_epochs):
        for x0 in dataloader:
          # CIFAR10의 약 50000개 dataset에 대해 batch size 64로 load
          # x0 : [64,3,H,W]
            x0 = x0.to(device)

            t = torch.randint(0, T, (x0.size(0),), device=device)
            # UNet 학습에 사용할 0과 T 사이의 랜덤 time_step 생성 
            noise = torch.randn_like(x0)
            # eps_pred와의 loss 계산에 사용할 noise 생성 
            a_bar = alpha_bar[t].view(-1, 1, 1, 1)
            # schedule 
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
            # schedule을 사용해 x0와 noise로부터 x_t 생성 

            eps_pred = model(x_t, t)
            # 1) UNet의 forward 실행 

            # 학습 
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 1) t -> t_emb를 만들어내는 nn.Linear와
            # 2) x_t -> eps_pred를 만들어내는 conv가 학습된다.
            #   여기서 t_emb는 unet의 각 conv 단계에 timestep 정보를 제공하는 역할을 한다
            #   그런데 t_emb를 각 layer의 feature map에 더해줌으로써 그 정보를 반영하는데,
            #   feature map의 모든 픽셀에 같은 수를 더해주면 relu로 인해 죽는 픽셀 개수가 줄어들고
            #   해당 feature map의 활성화 정도를 높이는 역할을 하게 된다.
            #   nn.Linear의 weight를 학습시켜서 timestep마다 어떤 feature map을 활성화시킬지 결정하고
            #   이는 단계에 따라 복원해야할 중요한 특징을 강화해주는 역할을 하게 된다 

        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

# --- Setup Dataset and Train ---
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

dataloader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
                        batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_unet(model, dataloader, optimizer, device)
