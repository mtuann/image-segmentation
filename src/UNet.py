import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self):
        print("Call Unet init")
        super().__init__()


    def forward(self, x):
        pass

def main():
    unet = Unet()

if __name__=="__main__":
    main()
