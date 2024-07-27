import torch
from net.ITA import JNet, TNet, GNet, TBNet
from net.mamba import JNet_mamba

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.Jnet = JNet()
        self.Jnet = JNet_mamba()
        # self.image_net = Mynet()
        # self.image_net = Decoder(device='cuda')
        # self.image_net = Unet()
        self.Tnet = TNet()
        self.TBnet = TBNet()

    def forward(self, data):
        x_j = self.Jnet(data)
        x_t = self.Tnet(data)
        x_tb = self.TBnet(data)
        # X_g = self.Gnet(data)
        # x_a = self.A_net(data)
        return x_j, x_t, x_tb
        # return x_j



