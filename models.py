import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, features, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(features, channels, 3, padding=1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        residual = self.layers(x)
        return x - residual

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024 // factor))
        
        self.up1 = nn.ConvTranspose2d(1024, 512 // factor, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512 // factor)
        self.up2 = nn.ConvTranspose2d(512, 256 // factor, 2, stride=2)
        self.conv2 = DoubleConv(512, 256 // factor)
        self.up3 = nn.ConvTranspose2d(256, 128 // factor, 2, stride=2)
        self.conv3 = DoubleConv(256, 128 // factor)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        return self.outc(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.ca = ChannelAttention(in_channels, reduction)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out) * out
        return out + residual

class RCAN(nn.Module):
    def __init__(self, n_channels=1, n_feats=64, n_blocks=10, reduction=16):
        super(RCAN, self).__init__()
        self.head = nn.Conv2d(n_channels, n_feats, 3, padding=1)
        
        self.body = nn.ModuleList([
            ResidualChannelAttentionBlock(n_feats, reduction) 
            for _ in range(n_blocks)
        ])
        
        self.tail = nn.Conv2d(n_feats, n_channels, 3, padding=1)
        
    def forward(self, x):
        x = self.head(x)
        residual = x
        
        for block in self.body:
            x = block(x)
        
        x = x + residual
        x = self.tail(x)
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class NAFNet(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class DRUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(DRUNet, self).__init__()

        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)

        self.m_down1 = self._make_layer(nc[0], nc[0], nb, act_mode)
        self.m_down2 = self._make_layer(nc[0], nc[1], nb, act_mode)
        self.m_down3 = self._make_layer(nc[1], nc[2], nb, act_mode)
        self.m_down4 = self._make_layer(nc[2], nc[3], nb, act_mode)

        self.m_body = self._make_layer(nc[3], nc[3], nb, act_mode)

        self.m_up4 = self._make_layer(nc[3]+nc[2], nc[2], nb, act_mode)
        self.m_up3 = self._make_layer(nc[2]+nc[1], nc[1], nb, act_mode)
        self.m_up2 = self._make_layer(nc[1]+nc[0], nc[0], nb, act_mode)
        self.m_up1 = self._make_layer(nc[0]+nc[0], nc[0], nb, act_mode)

        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

        if downsample_mode == 'avgpool':
            downsample_block = nn.AvgPool2d
        elif downsample_mode == 'maxpool':
            downsample_block = nn.MaxPool2d
        elif downsample_mode == 'strideconv':
            downsample_block = lambda x: nn.Conv2d(x, x, kernel_size=2, stride=2, padding=0, bias=False)

        if upsample_mode == 'upconv':
            upsample_block = lambda x: nn.ConvTranspose2d(x, x, kernel_size=2, stride=2, padding=0, bias=False)
        elif upsample_mode == 'pixelshuffle':
            upsample_block = lambda x: nn.Sequential(nn.Conv2d(x, 4*x, 3, 1, 1, bias=False), nn.PixelShuffle(2))
        elif upsample_mode == 'convtranspose':
            upsample_block = lambda x: nn.ConvTranspose2d(x, x, kernel_size=2, stride=2, padding=0, bias=False)

        self.downsample1 = downsample_block(nc[0])
        self.downsample2 = downsample_block(nc[1])
        self.downsample3 = downsample_block(nc[2])
        self.upsample3 = upsample_block(nc[3])
        self.upsample2 = upsample_block(nc[2])
        self.upsample1 = upsample_block(nc[1])

    def _make_layer(self, in_channels, out_channels, nb, act_mode='R'):
        layers = []
        for i in range(nb):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1, bias=False))
            if act_mode == 'R':
                layers.append(nn.ReLU(inplace=True))
            elif act_mode == 'L':
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x2 = self.downsample1(x2)
        x3 = self.m_down2(x2)
        x3 = self.downsample2(x3)
        x4 = self.m_down3(x3)
        x4 = self.downsample3(x4)
        x5 = self.m_down4(x4)

        x5 = self.m_body(x5)

        x = self.m_up4(torch.cat([x4, self.upsample3(x5)], 1))
        x = self.m_up3(torch.cat([x3, self.upsample2(x)], 1))
        x = self.m_up2(torch.cat([x2, self.upsample1(x)], 1))
        x = self.m_up1(torch.cat([x1, x], 1))

        x = self.m_tail(x)
        return x

class FFDNet(nn.Module):
    def __init__(self, num_input_channels=1, num_feature_maps=64, num_layers=15):
        super(FFDNet, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_feature_maps = num_feature_maps
        self.num_layers = num_layers
        
        layers = []
        layers.append(nn.Conv2d(num_input_channels + 1, num_feature_maps, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_feature_maps, num_feature_maps, 3, padding=1))
            layers.append(nn.BatchNorm2d(num_feature_maps))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(num_feature_maps, num_input_channels, 3, padding=1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x, noise_sigma):
        noise_map = noise_sigma.expand_as(x[:, :1, :, :])
        x_in = torch.cat([x, noise_map], dim=1)
        residual = self.layers(x_in)
        return x - residual


class RIDNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_channels=64, num_blocks=4):
        super(RIDNet, self).__init__()
        
        self.head = nn.Conv2d(in_channels, feature_channels, 3, padding=1)
        
        self.body = nn.ModuleList([
            ResidualInResidualDenseBlock(feature_channels) 
            for _ in range(num_blocks)
        ])
        
        self.tail = nn.Sequential(
            nn.Conv2d(feature_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.head(x)
        residual = x
        
        for block in self.body:
            x = block(x)
        
        x = x + residual
        x = self.tail(x)
        return x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super(ResidualInResidualDenseBlock, self).__init__()
        
        self.rdb1 = ResidualDenseBlock(channels, growth_rate)
        self.rdb2 = ResidualDenseBlock(channels, growth_rate)
        self.rdb3 = ResidualDenseBlock(channels, growth_rate)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, padding=1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class BRDNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=20):
        super(BRDNet, self).__init__()
        
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        
        self.body = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.BatchNorm2d(num_features)
            ) for _ in range(num_blocks)
        ])
        
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)
        
    def forward(self, x):
        x = self.head(x)
        residual = x
        
        for block in self.body:
            x = F.relu(block(x) + x)
        
        x = x + residual
        x = self.tail(x)
        return x


class HINet(nn.Module):
    def __init__(self, in_chn=3, wf=64, depth=5):
        super(HINet, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        
        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample))
            prev_channels = (2**i) * wf
            
        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
            
        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = nn.Conv2d(prev_channels, in_chn, 3, 1, 1)
        
    def get_input_chn(self, wf):
        return wf
        
    def forward(self, x):
        image = x
        x1 = self.conv_01(image)
        x2 = self.conv_02(image)
        blocks_1 = []
        blocks_2 = []
        
        for i, (down1, down2) in enumerate(zip(self.down_path_1, self.down_path_2)):
            if i != len(self.down_path_1)-1:
                blocks_1.append(x1)
                blocks_2.append(x2)
            x1 = down1(x1)
            x2 = down2(x2)
            
        for i, (up1, up2) in enumerate(zip(self.up_path_1, self.up_path_2)):
            x1 = up1(x1, self.skip_conv_1[i](blocks_1[-i-1]))
            x2 = up2(x2, self.skip_conv_2[i](blocks_2[-i-1]))
            
        x1 = self.sam12(x1, x2)
        x_cat = torch.cat([x1, x2], 1)
        x_cat = self.cat12(x_cat)
        x = self.last(x_cat) + image
        
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))
        
        if downsample:
            self.downsample = nn.Conv2d(out_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)
            
    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False)
        
    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, 3, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv3 = nn.Conv2d(3, n_feat, kernel_size, padding=kernel_size//2, bias=bias)
        
    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1