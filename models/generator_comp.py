try:
    from models.conformer import ConformerBlock
except ImportError:  # pragma: no cover - supports direct module execution
    from .conformer import ConformerBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels, kernel_size, r=1, padding=(1, 1, 0, 0)):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d(padding=padding, value=0.0)
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


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x


class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)
        self.dense_R = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.dense_P = DilatedDenseNet(depth=4, in_channels=num_channel)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(
            num_channel=num_channel
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        #x.unsqueeze(dim=1)
        x = torch.swapaxes(x, 3, 2)
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        real_in = self.dense_R(out_5)
        imag_in = self.dense_P(out_5)

        feature_0 = self.avg_pool(real_in)
        feature_1 = self.avg_pool(imag_in)
        feature_2 = self.avg_pool(out_1)
        feature_3 = self.avg_pool(out_2)
        feature_4 = self.avg_pool(out_3)
        feature_5 = self.avg_pool(out_4)
        feature_6 = self.avg_pool(out_5)

        mask = self.mask_decoder(real_in)
        out_mag = mask * mag
        complex_out = self.complex_decoder(imag_in)

        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        # feature_out = torch.concat((feature_0, feature_1, feature_2, feature_3, feature_4), dim=1)[:, :, 0, 0]
        feature_out = torch.concat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6), dim=1)[:, :, 0, 0]
        # feature_out = torch.concat((feature_0, feature_1), dim=1)[:, :, 0, 0]
        # feature_out = torch.concat((feature_0, feature_1, feature_2), dim=1)[:, :, 0, 0]
        final_out = torch.concat((final_real, final_imag), dim=1)
        final_out = torch.swapaxes(final_out, 3, 2)
        return final_out, feature_out


class TSCNetnonCB(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(TSCNetnonCB, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        # self.TSCB_1 = TSCB(num_channel=num_channel)
        # self.TSCB_2 = TSCB(num_channel=num_channel)
        # self.TSCB_3 = TSCB(num_channel=num_channel)
        # self.TSCB_4 = TSCB(num_channel=num_channel)
        self.dense_R = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.dense_P = DilatedDenseNet(depth=4, in_channels=num_channel)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(
            num_channel=num_channel
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        #x.unsqueeze(dim=1)
        x = torch.swapaxes(x, 3, 2)
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        # out_2 = self.TSCB_1(out_1)
        # out_3 = self.TSCB_2(out_2)
        # out_4 = self.TSCB_3(out_3)
        # out_5 = self.TSCB_4(out_4)

        real_in = self.dense_R(out_1)
        imag_in = self.dense_P(out_1)

        feature_0 = self.avg_pool(real_in)
        feature_1 = self.avg_pool(imag_in)
        feature_2 = self.avg_pool(out_1)
        # feature_3 = self.avg_pool(out_2)
        # feature_4 = self.avg_pool(out_3)
        # feature_5 = self.avg_pool(out_4)
        # feature_6 = self.avg_pool(out_5)

        mask = self.mask_decoder(real_in)
        out_mag = mask * mag
        complex_out = self.complex_decoder(imag_in)

        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        # feature_out = torch.concat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6), dim=1)[:, :, 0, 0]
        # feature_out = torch.concat((feature_0, feature_1, feature_2, feature_3, feature_4), dim=1)[:, :, 0, 0]
        feature_out = torch.concat((feature_0, feature_1, feature_2), dim=1)[:, :, 0, 0]
        final_out = torch.concat((final_real, final_imag), dim=1)
        final_out = torch.swapaxes(final_out, 3, 2)
        return final_out, feature_out


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


class TSCNet_Cont(nn.Module):
    def __init__(self, num_channel=64, num_features=201, cb=0):
        super(TSCNet_Cont, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        if cb == 4:
            self.TSCB_1 = TSCB(num_channel=num_channel)
            self.TSCB_2 = TSCB(num_channel=num_channel)
            self.TSCB_3 = TSCB(num_channel=num_channel)
            self.TSCB_4 = TSCB(num_channel=num_channel)
        elif cb == 3:
            self.TSCB_1 = TSCB(num_channel=num_channel)
            self.TSCB_2 = TSCB(num_channel=num_channel)
            self.TSCB_3 = TSCB(num_channel=num_channel)
        elif cb == 2:
            self.TSCB_1 = TSCB(num_channel=num_channel)
            self.TSCB_2 = TSCB(num_channel=num_channel)
        elif cb == 1:
            self.TSCB_1 = TSCB(num_channel=num_channel)
        elif cb == 0:
            pass
        else:
            raise ValueError(f"Unsupported cb={cb}. Expected 0, 1, 2, 3, or 4.")
        self.dense_R = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.dense_P = DilatedDenseNet(depth=4, in_channels=num_channel)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(
            num_channel=num_channel
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cb = cb

    def forward(self, x):
        #x.unsqueeze(dim=1)
        x = torch.swapaxes(x, 3, 2)
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        if self.cb == 4:
            out_2 = self.TSCB_1(out_1)
            out_3 = self.TSCB_2(out_2)
            out_4 = self.TSCB_3(out_3)
            out_5 = self.TSCB_4(out_4)

            real_in = self.dense_R(out_5)
            imag_in = self.dense_P(out_5)
        elif self.cb == 3:
            out_2 = self.TSCB_1(out_1)
            out_3 = self.TSCB_2(out_2)
            out_4 = self.TSCB_3(out_3)

            real_in = self.dense_R(out_4)
            imag_in = self.dense_P(out_4)
        elif self.cb == 2:
            out_2 = self.TSCB_1(out_1)
            out_3 = self.TSCB_2(out_2)
            real_in = self.dense_R(out_3)
            imag_in = self.dense_P(out_3)
        elif self.cb == 1:
            out_2 = self.TSCB_1(out_1)
            real_in = self.dense_R(out_2)
            imag_in = self.dense_P(out_2)
        else:
            real_in = self.dense_R(out_1)
            imag_in = self.dense_P(out_1)

        feature_0 = self.avg_pool(real_in)
        feature_1 = self.avg_pool(imag_in)

        mask = self.mask_decoder(real_in)
        out_mag = mask * mag
        complex_out = self.complex_decoder(imag_in)

        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        if self.cb == 4:
            feature_2 = self.avg_pool(out_2)
            feature_3 = self.avg_pool(out_3)
            feature_4 = self.avg_pool(out_4)
            feature_5 = self.avg_pool(out_5)
            feature_out = torch.concat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5), dim=1)[:, :, 0, 0]
        elif self.cb == 3:
            feature_2 = self.avg_pool(out_2)
            feature_3 = self.avg_pool(out_3)
            feature_4 = self.avg_pool(out_4)
            feature_out = torch.concat((feature_0, feature_1, feature_2, feature_3, feature_4), dim=1)[:, :, 0, 0]
        elif self.cb == 2:
            feature_2 = self.avg_pool(out_2)
            feature_3 = self.avg_pool(out_3)
            feature_out = torch.concat((feature_0, feature_1, feature_2, feature_3), dim=1)[:, :, 0, 0]
        elif self.cb == 1:
            feature_2 = self.avg_pool(out_2)
            feature_out = torch.concat((feature_0, feature_1, feature_2), dim=1)[:, :, 0, 0]
        else:
            feature_out = torch.concat((feature_0, feature_1), dim=1)[:, :, 0, 0]
        final_out = torch.concat((final_real, final_imag), dim=1)
        final_out = torch.swapaxes(final_out, 3, 2)
        return final_out, feature_out

# model = TSCNetCPCLF(num_channel=40, num_features=201)
# summary(model, [1, 1, 201, 534])


class ArcFaceClassifier(nn.Module):
    def __init__(self, emb_size, output_classes, gpu_id):
        super().__init__()
        self.W = nn.Parameter(torch.empty(emb_size, output_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        # Step 1:
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        # Step 2:
        return x_norm @ W_norm


def arcface_loss(cosine, targ, output_classes, s=30.0, m=.4):
    # this prevents nan when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1 + 1e-7, 1 - 1e-7)
    # Step 3:
    arcosine = cosine.arccos()
    # Step 4:
    arcosine += F.one_hot(targ, num_classes=output_classes) * m
    # Step 5:
    cosine2 = arcosine.cos()
    # Step 6:

    scaled_cosine = cosine2 * s
    return F.cross_entropy(scaled_cosine, targ)
