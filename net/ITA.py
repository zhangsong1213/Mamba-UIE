
import torch
import torch.nn as nn
from torch.nn import Parameter
from switchable_norm import SwitchNorm2d
from torchvision.models.vgg import vgg16
from torch.distributions import kl
from utils1 import  ResBlock, ConvBlock, Up, Compute_z, PixelShuffleUpsample
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()

def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6

def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)

class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


def _make_divisible(v, divisor=8, min_value=None):  ## 将通道数变成8的整数倍
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                Mish(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1).cuda()
        return x * y



class TNet(torch.nn.Module):  ## 加宽加深的
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, 64, 1, 1, 0),
            # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
            torch.nn.InstanceNorm2d(64),
            # torch.nn.ReLU()
            Mish()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 64, 3, 1, 0),
            torch.nn.InstanceNorm2d(64),
            # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
            Mish(),
            SELayer(64, 64)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.conv5(data)
        data1 = self.final(data)

        return data1


class TBNet(torch.nn.Module):  ## 加宽加深的
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, 64, 1, 1, 0),
            # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
            torch.nn.InstanceNorm2d(64),
            # torch.nn.ReLU()
            Mish()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 64, 3, 1, 0),
            torch.nn.InstanceNorm2d(64),
            # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
            Mish(),
            SELayer(64, 64)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.conv5(data)
        data1 = self.final(data)

        return data1

class JNet(torch.nn.Module):  ## 加宽加深的
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, 64, 1, 1, 0),
            # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
            torch.nn.InstanceNorm2d(64),
            # torch.nn.ReLU()
            Mish()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, 3, 1, 0),
            torch.nn.InstanceNorm2d(256),
            # SwitchNorm2d(256, using_moving_average=False, using_bn=True),
            Mish()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 64, 3, 1, 0),
            torch.nn.InstanceNorm2d(64),
            # SwitchNorm2d(64, using_moving_average=False, using_bn=True),
            Mish(),
            SELayer(64, 64)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.conv5(data)
        data1 = self.final(data)

        return data1

class GNet(nn.Module):  ## 用于估算卷积核
    def __init__(self):
        super(GNet, self).__init__()
        # 定义生成 g_out 的卷积层，可以根据需要调整
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64, 3 * 3 * 9 * 9)  # 全连接层，用于输出期望形状

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.mean([2, 3])
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        g_out = x.view(3, 3, 9, 9)
        return g_out


# if __name__ == "__main__":
#     model = MambaLayer(dim=3).cuda()
#     input = torch.zeros((2, 1, 128, 128)).cuda()  ##
#     input1 = torch.randn(1, 3, 256, 256).cuda()
#     output = model(input1)
#     print(output.shape)

