#!/usr/bin/env python3
import torch, torch.nn as nn
from PIL import Image
import torchvision.transforms as T
class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.dec = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(64,32,3,padding=1),nn.ReLU(),nn.Upsample(scale_factor=2),nn.Conv2d(32,1,3,padding=1),nn.Sigmoid())
    def forward(self,x): return self.dec(self.enc(x))
class BGRemover:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegNet().to(self.dev).eval()
        self.tf = T.Compose([T.Resize((512,512)),T.ToTensor()])
    def remove(self,inp,out):
        img = Image.open(inp).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.dev)
        with torch.no_grad(): mask = self.model(x).squeeze().cpu().numpy()
        mask_img = Image.fromarray((mask*255).astype("uint8")).resize(img.size)
        result = img.copy(); result.putalpha(mask_img); result.save(out)
        print(f"Saved {out}")
if __name__=="__main__":
    import sys; BGRemover().remove(sys.argv[1],"output_nobg.png")
