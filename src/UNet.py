# /data.local/tuannm/Git-Code/image-segmentation/src/UNet.py
# src: https://github.com/milesial/Pytorch-UNet
from Layers import *

from torchsummary import summary


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def forward_debug(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)        
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        print("x: {}\nx1: {}\nx2: {}\nx3: {}\nx4: {}\nx5: {}".format(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape))
        x = self.up1(x5, x4)
        print("up1 x5 - x4: {}".format(x.shape))
        
        x = self.up2(x, x3)
        print("up2 x - x3: {}".format(x.shape))
        
        x = self.up3(x, x2)
        print("up3 x - x2: {}".format(x.shape))
        
        x = self.up4(x, x1)
        print("up4 x - x1: {}".format(x.shape))
        
        logits = self.outc(x)
        print("logits: {}".format(logits.shape))
        return logits
    
#     def model_print(self):
#         for p in self.children():
#             print(p.shape)
    
def main():
    unet = UNet(n_channels=3, n_classes=1, bilinear=True)
#     unet.model_print()
#     pytorch_total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
#     print(pytorch_total_params)
#     print(unet)
#     summary(unet, (3, 120, 120))
    x = torch.rand(10, 3, 636, 434)
    output = unet(x)
    
if __name__=="__main__":
    main()
