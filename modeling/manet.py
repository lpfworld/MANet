import torch
import torch.nn as nn
import torch.nn.functional as F

#　1-DDCB-model
class DDCB(nn.Module):
    def __init__(self, in_planes):
        super(DDCB, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, 256, 1), nn.ReLU(True), nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes+64, 256, 1), nn.ReLU(True), nn.Conv2d(256, 64, 3, padding=2, dilation=2), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes+128, 256, 1), nn.ReLU(True), nn.Conv2d(256, 64, 3, padding=3, dilation=3), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_planes+128, 512, 3, padding=1), nn.ReLU(True))
    def forward(self, x):
        x1_raw = self.conv1(x)
        x1 = torch.cat([x, x1_raw], 1)
        x2_raw = self.conv2(x1)
        x2 = torch.cat([x, x1_raw, x2_raw], 1)
        x3_raw = self.conv3(x2)
        x3 = torch.cat([x, x2_raw, x3_raw], 1)
        output = self.conv4(x3)
        return output

# 2-PyConv-model
def ConvBNReLU(in_channels,out_channels,kernel_size,stride,groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=kernel_size//2,groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def Conv1x1BNReLU(in_channels,out_channels,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class PyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, groups, stride=1):
        super(PyConv, self).__init__()
        if out_channels is None:
            out_channels = []
        assert len(out_channels) == len(kernel_sizes) == len(groups)

        self.pyconv_list = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.pyconv_list.append(ConvBNReLU(in_channels=in_channels,out_channels=out_channels[i],kernel_size=kernel_sizes[i],stride=stride,groups=groups[i]))

    def forward(self, x):
        outputs = []
        for pyconv in self.pyconv_list:
            outputs.append(pyconv(x))
        return torch.cat(outputs, 1)

class LocalPyConv(nn.Module):
    def __init__(self, planes):
        super(LocalPyConv, self).__init__()
        inplanes = planes//4
        self._reduce = Conv1x1BNReLU(planes, 512)
        self._pyConv = PyConv(in_channels=512, out_channels=[inplanes, inplanes, inplanes, inplanes], kernel_sizes=[3, 5, 7, 9], groups=[1, 4, 8, 16])
        self._combine = Conv1x1BNReLU(512, planes)

    def forward(self, x):
        return self._combine(self._pyConv(self._reduce(x)))

class GlobalPyConv(nn.Module):
    def __init__(self, planes):
        super(GlobalPyConv, self).__init__()
        inplanes = planes // 4
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=9)
        self._reduce = Conv1x1BNReLU(planes, 512)
        self._pyConv = PyConv(in_channels=512, out_channels=[inplanes, inplanes, inplanes, inplanes],
                              kernel_sizes=[3, 5, 7, 9], groups=[1, 4, 8, 16])
        self._fuse = Conv1x1BNReLU(512, 512)

    def forward(self, x):
        b,c,w,h = x.shape
        x = self._fuse(self._pyConv(self._reduce(self.global_pool(x))))
        out = F.interpolate(x,(w,h),align_corners=True,mode='bilinear')
        return out

class MergePyConv(nn.Module):
    def __init__(self, in_channels):
        super(MergePyConv, self).__init__()
        #self.img_size = img_size
        self.conv3 = ConvBNReLU(in_channels=in_channels,out_channels=256,kernel_size=3,stride=1)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1,groups=1)

    def forward(self, x):
        x = self.conv3(x)
        #x = F.interpolate(x, self.img_size, align_corners=True,mode='bilinear')
        #下采样,上采样函数
        out = self.conv1(x)
        return out

class PyConvParsingHead(nn.Module):
    def __init__(self, planes):
        super(PyConvParsingHead, self).__init__()

        self.globalPyConv = GlobalPyConv(planes=planes)
        self.localPyConv = LocalPyConv(planes=planes)
        self.mergePyConv = MergePyConv(1024)

    def forward(self, x):
        g_x = self.globalPyConv(x)
        l_x = self.localPyConv(x)
        x = torch.cat([g_x,l_x],dim=1)
        out = self.mergePyConv(x)
        return out


# 3-CSPN-model
class CBR(nn.Module):

    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        # self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        # self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input

class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


# denseCon
class DenseScaleNet(nn.Module):
    def __init__(self, load_model=''):
        super(DenseScaleNet, self).__init__()
        self.load_model = load_model
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,]
        self.features = make_layers(self.features_cfg)
        self.DDCB1 = DilatedParllelResidualBlockB(512, 512)
        self.DDCB2 = DilatedParllelResidualBlockB(512, 512)
        self.DDCB3 = DilatedParllelResidualBlockB(512, 512)

        self.output_layers = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(True), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1))
        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x1_raw = self.DDCB1(x)
        print(x.shape,x1_raw.shape)
        x1 = x1_raw + x
        print(x1.shape)
        x2_raw = self.DDCB2(x1)
        x2 = x2_raw + x1_raw + x
        x3_raw = self.DDCB3(x2)
        x3 = x3_raw + x2_raw + x1_raw + x
        output = self.output_layers(x3)
        return output
    def _initialize_weights(self):
        self_dict = self.state_dict()
        pretrained_dict = dict()
        self._random_initialize_weights()
        if not self.load_model:
            vgg16 = torch.load("/home/lpf/PycharmProjects/Dense-Scale-Network-for-Crowd-Counting-master/vgg16-397923af.pth")
            for k, v in vgg16.items():
                if k in self_dict and self_dict[k].size() == v.size():
                    pretrained_dict[k] = v
            self_dict.update(pretrained_dict)
            self.load_state_dict(self_dict)
        else:
            self.load_state_dict(torch.load(self.load_model))
    def _random_initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                #nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

