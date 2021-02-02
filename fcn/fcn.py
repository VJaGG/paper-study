import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# 用双线性插值的方式初始化，确实可以加快训练的速度
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg_feature = torch.hub.load('pytorch/vision',
                                     'vgg16_bn', pretrained=False).features
        self.encode1 = vgg_feature[0: 7]
        self.encode2 = vgg_feature[7: 14]
        self.encode3 = vgg_feature[14: 24]
        self.encode4 = vgg_feature[24: 34]
        self.encode5 = vgg_feature[34:]

        self.conv1 = nn.Conv2d(512, 4096, 7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(4096)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d()
        self.conv5 = nn.Conv2d(4096, 4096, 1, stride=1)
        self.bn2 = nn.BatchNorm2d(4096)
        self.dropout1 = nn.Dropout2d()

        # weight init?
        self.conv2 = nn.Conv2d(4096, num_classes, 1)
        self.upsample1 = nn.ConvTranspose2d(num_classes,
                                            num_classes,
                                            4, 2, 1, bias=False)
        self.upsample1.weight.data = bilinear_kernel(num_classes,
                                                     num_classes, 4)

        self.conv6 = nn.Conv2d(512, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.dropout2 = nn.Dropout2d()

        self.conv3 = nn.Conv2d(512, num_classes, 1)
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes,
                                            4, 2, 1, bias=False)
        self.upsample2.weight.data = bilinear_kernel(num_classes,
                                                     num_classes, 4)

        self.conv4 = nn.Conv2d(256, num_classes, 1)
        self.upsample3 = nn.ConvTranspose2d(num_classes, num_classes,
                                            16, 8, 4, bias=False)
        self.upsample3.weight.data = bilinear_kernel(num_classes,
                                                     num_classes, 16)

    def forward(self, x):
        x = self.encode1(x)
        x = self.encode2(x)
        feature_8x = self.encode3(x)  # [1, 256, 32, 32]
        feature_16x = self.encode4(feature_8x)  # [1, 512, 16, 16]
        feature_32x = self.encode5(feature_16x)  # [1, 512, 8, 8]
        feature_32x = self.conv1(feature_32x)  # [1, 1024, 8, 8]
        feature_32x = self.bn1(feature_32x)
        feature_32x = self.relu(feature_32x)
        feature_32x = self.dropout(feature_32x)

        feature_32x = self.conv5(feature_32x)
        feature_32x = self.bn2(feature_32x)
        feature_32x = self.relu(feature_32x)
        feature_32x = self.dropout1(feature_32x)

        feature_32x = self.conv2(feature_32x)
        feature_32x = self.upsample1(feature_32x)

        feature_16x = self.conv3(feature_16x)
        feature_16x = feature_16x + feature_32x

        feature_16x = self.upsample2(feature_16x)
        feature_8x = self.conv4(feature_8x)
        feature_8x = feature_8x + feature_16x
        output = self.upsample3(feature_8x)
        return output


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = FCN(21)
    from torchsummary import summary
    summary(model, input_size=(3, 256, 256), device="cpu")
