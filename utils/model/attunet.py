import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        use_attention: bool = False,
        use_res: bool = False,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_attention = use_attention
        self.use_res = use_res

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, use_res)])
        if use_attention:
            self.down_att_layers = nn.ModuleList([AttentionBlock(chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob, use_res))
            if use_attention:
                self.down_att_layers.append(AttentionBlock(ch * 2))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, drop_prob, use_res)
        if use_attention:
            self.conv_att = AttentionBlock(ch * 2)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        if use_attention:
            self.up_att = nn.ModuleList()
        for _ in range(num_pool_layers):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob, use_res))
            if use_attention:
                self.up_att.append(AttentionBlock(ch))
            ch //= 2

        self.out_conv = nn.Conv2d(ch * 2, self.out_chans, kernel_size=1, stride=1)
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image
        print("input: ", image.shape)
        if self.use_attention:  # use attention
            # apply down-sampling layers
            for layer, att in zip(self.down_sample_layers, self.down_att_layers):
                output = layer(output)
                output = att(output)
                stack.append(output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

            output = self.conv(output)
            output = self.conv_att(output)
# #             print("downsampled: ", output.shape)
            # apply up-sampling layers
            for transpose_conv, conv, att in zip(self.up_transpose_conv, self.up_conv, self.up_att):
                downsample_layer = stack.pop()
                output = transpose_conv(output)
#                 print("transposed: ", output.shape)
                # reflect pad on the right/botton if needed to handle odd input dimensions
                padding = [0, 0, 0, 0]
                if output.shape[-1] != downsample_layer.shape[-1]:
                    padding[1] = 1  # padding right
                if output.shape[-2] != downsample_layer.shape[-2]:
                    padding[3] = 1  # padding bottom
                if torch.sum(torch.tensor(padding)) != 0:
                    output = F.pad(output, padding, "reflect")

                output = torch.cat([output, downsample_layer], dim=1)
                output = conv(output)
                output = att(output)
            output = self.out_conv(output)
            print(output.shape)

        else:  # no attention
            # apply down-sampling layers
            for layer in self.down_sample_layers:
                output = layer(output)
                stack.append(output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

            output = self.conv(output)

            # apply up-sampling layers
            for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
                downsample_layer = stack.pop()
                output = transpose_conv(output)

                # reflect pad on the right/botton if needed to handle odd input dimensions
                padding = [0, 0, 0, 0]
                if output.shape[-1] != downsample_layer.shape[-1]:
                    padding[1] = 1  # padding right
                if output.shape[-2] != downsample_layer.shape[-2]:
                    padding[3] = 1  # padding bottom
                if torch.sum(torch.tensor(padding)) != 0:
                    output = F.pad(output, padding, "reflect")

                output = torch.cat([output, downsample_layer], dim=1)
                output = conv(output)
            output = self.out_conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            drop_prob: float,
            use_res: bool = True,):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.use_res = use_res

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_chans),
        )

        self.layers_out = nn.Sequential(
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        if self.use_res:
            return self.layers_out(self.layers(image) + self.conv1x1(image))
        else:
            return self.layers_out(self.layers(image))


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class AttentionBlock(nn.Module):
    """
    Attention block with channel and spatial-wise attention mechanism.
    """
    def __init__(self, num_ch, r=2):
        super(AttentionBlock, self).__init__()
        self.C = num_ch
        self.r = r

        self.sig = nn.Sigmoid()
        # channel attention
        self.fc_ch = nn.Sequential(nn.Linear(self.C, self.C//self.r),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.C//self.r, self.C),)
        # spatial attention
        self.conv = nn.Conv2d(self.C, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, inputs):  # [N,C,H,W]
        b, c, h, w = inputs.shape
        # spatial attention
        sa = self.conv(inputs)
        sa = self.sig(sa)
        inputs_s = sa * inputs

        # channel attention
        ca = torch.abs(inputs)
        # ca = self.pool(ca)  # [B,C,1,1]
        ca = torch.mean(ca.reshape(b, c, -1), dim=2)  # [B,C]
        ca = self.fc_ch(ca)  # [B,C]
        ca = self.sig(ca).reshape(b, c, 1, 1)  #[B,C,1,1]
        inputs_c = ca * inputs

        outputs = torch.max(inputs_s, inputs_c)
        return outputs