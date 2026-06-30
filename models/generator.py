try:
    from models.conformer import ConformerBlock
except ImportError:  # pragma: no cover - supports direct module execution
    from .conformer import ConformerBlock
import torch
import torch.nn as nn


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Decoder0(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(Decoder0, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201, batch_size=1):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=1, channels=num_channel)

        # self.TSCB_1 = TSCB(num_channel=num_channel)
        # self.TSCB_2 = TSCB(num_channel=num_channel)
        # self.TSCB_3 = TSCB(num_channel=num_channel)
        # self.TSCB_4 = TSCB(num_channel=num_channel)

        self.decoder0 = Decoder0(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.decoder1 = Decoder0(
            num_features, num_channel=num_channel, out_channel=1
        )

    def forward(self, x):
        #x.unsqueeze(dim=1)
        x = torch.swapaxes(x.unsqueeze(dim=1), 3, 2)
        out_3 = self.dense_encoder(x)
        # out_3 = self.TSCB_1(out_1)
        #out_3 = self.TSCB_2(out_2)
        # out_4 = self.TSCB_3(out_3)
        # out_5 = self.TSCB_4(out_4)

        decoder_0 = torch.swapaxes(self.decoder0(out_3), 3, 2).squeeze(dim=1)
        decoder_1 = torch.swapaxes(self.decoder1(out_3), 3, 2).squeeze(dim=1)

        max_c = torch.nn.functional.max_pool2d(
            out_3, (out_3.size()[2], out_3.size()[3])).squeeze(dim=3).squeeze(dim=2)
        max_w = out_3.max(dim=1)[0].max(dim=1)[0]
        max_h = out_3.max(dim=1)[0].max(dim=2)[0]
        feature_out = torch.concat([max_c, max_w, max_h], dim=1)

        return decoder_0, decoder_1, feature_out


class TSCNetMP(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(TSCNetMP, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=1, channels=num_channel)

        self.decoder0 = Decoder0(
            num_features, num_channel=num_channel, out_channel=1
        )

    def forward(self, x):
        #x.unsqueeze(dim=1)
        # x = torch.swapaxes(x.unsqueeze(dim=1), 3, 2)
        x = torch.swapaxes(x, 3, 2)
        out_3 = self.dense_encoder(x)
        decoder_0 = torch.swapaxes(self.decoder0(out_3), 3, 2).squeeze(dim=1)

        max_c = torch.nn.functional.avg_pool2d(
            out_3, (out_3.size()[2], out_3.size()[3])).squeeze(dim=3).squeeze(dim=2)
        max_w = out_3.mean(dim=1).mean(dim=2)
        max_h = out_3.mean(dim=1).mean(dim=1)
        feature_out = torch.concat([max_c, max_w, max_h], dim=1)

        return decoder_0, feature_out


class TSCNet_enc(nn.Module):
    def __init__(self, num_channel=64, num_features=201, decoder=1):
        super(TSCNet_enc, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=1, channels=num_channel)


    def forward(self, x):
        #x.unsqueeze(dim=1)
        x = torch.swapaxes(x.unsqueeze(dim=1), 3, 2)
        out_3 = self.dense_encoder(x)

        max_c = torch.nn.functional.max_pool2d(
            out_3, (out_3.size()[2], out_3.size()[3])).squeeze(dim=3).squeeze(dim=2)
        max_w = out_3.max(dim=1)[0].max(dim=1)[0]
        max_h = out_3.max(dim=1)[0].max(dim=2)[0]
        feature_out = torch.concat([max_c, max_w, max_h], dim=1)

        return out_3, feature_out


class TSCNet_dec(nn.Module):
    def __init__(self, num_channel=64, num_features=201, decoder=1):
        super(TSCNet_dec, self).__init__()
        self.decoder0 = Decoder0(
            num_features, num_channel=num_channel, out_channel=1
        )
        if not decoder == 1:
            self.decoder1 = Decoder0(
                num_features, num_channel=num_channel, out_channel=1
            )
        self.decoder_flag = decoder

    def forward(self, x):

        decoder_0 = torch.swapaxes(self.decoder0(x), 3, 2).squeeze(dim=1)

        if not self.decoder_flag == 1:
            decoder_1 = torch.swapaxes(self.decoder1(x), 3, 2).squeeze(dim=1)

        if self.decoder_flag == 1:
            return decoder_0
        else:
            return decoder_0, decoder_1


class TSCNet_enc_p(nn.Module):
    def __init__(self, num_channel=64, pooling='dual'):
        super(TSCNet_enc_p, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=1, channels=num_channel)
        self.channel_num = num_channel
        self.pooling = pooling

    def forward(self, x):
        #x.unsqueeze(dim=1)
        x = torch.swapaxes(x.unsqueeze(dim=1), 3, 2)
        out_3 = self.dense_encoder(x)
        if self.pooling == 'avg':
            avg_c = nn.AdaptiveAvgPool2d(1)(out_3).squeeze(dim=(2, 3))
            avg_w = nn.AdaptiveAvgPool2d((out_3.size()[3], 1))(out_3).squeeze(dim=3).mean(dim=1)
            avg_h = nn.AdaptiveAvgPool2d((1, out_3.size()[2]))(out_3).squeeze(dim=2).mean(dim=1)

            feature_out = torch.concat([avg_c, avg_w, avg_h], dim=1)
        elif self.pooling == 'max':
            max_c = torch.nn.functional.max_pool2d(
                out_3, (out_3.size()[2], out_3.size()[3])).squeeze(dim=3).squeeze(dim=2)
            max_w = out_3.max(dim=1)[0].max(dim=1)[0]
            max_h = out_3.max(dim=1)[0].max(dim=2)[0]
            feature_out = torch.concat([max_c, max_w, max_h], dim=1)
        elif self.pooling == 'dual':
            avg_c = nn.AdaptiveAvgPool2d(1)(out_3).squeeze(dim=(2, 3))
            avg_w = nn.AdaptiveAvgPool2d((out_3.size()[3], 1))(out_3).squeeze(dim=3).mean(dim=1)
            avg_h = nn.AdaptiveAvgPool2d((1, out_3.size()[2]))(out_3).squeeze(dim=2).mean(dim=1)

            max_c = torch.nn.functional.max_pool2d(
                out_3, (out_3.size()[2], out_3.size()[3])).squeeze(dim=3).squeeze(dim=2)
            max_w = out_3.max(dim=1)[0].max(dim=1)[0]
            max_h = out_3.max(dim=1)[0].max(dim=2)[0]

            feature_out = torch.concat([avg_c, avg_w, avg_h, max_c, max_w, max_h], dim=1)

        return out_3, feature_out


class CLF_CONT(nn.Module):
    def __init__(self, num_channel=64, num_features=201, decoder=1):
        super(TSCNet_enc, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=1, channels=num_channel)


    def forward(self, x):
        #x.unsqueeze(dim=1)
        x = torch.swapaxes(x.unsqueeze(dim=1), 3, 2)
        out_3 = self.dense_encoder(x)

        max_c = torch.nn.functional.max_pool2d(
            out_3, (out_3.size()[2], out_3.size()[3])).squeeze(dim=3).squeeze(dim=2)
        max_w = out_3.max(dim=1)[0].max(dim=1)[0]
        max_h = out_3.max(dim=1)[0].max(dim=2)[0]
        feature_out = torch.concat([max_c, max_w, max_h], dim=1)

        return out_3, feature_out
