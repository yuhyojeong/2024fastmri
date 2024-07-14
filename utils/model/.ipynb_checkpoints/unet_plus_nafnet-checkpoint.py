from unet import Unet
from nafnet import NAFnet

class nafunet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.unet = Unet(
            in_chans = in_chans,
            out_chans = out_chans,
            chan= chans,
            num_pool_layers= num_pools,
            drop_prob= drop_prob,
        )
        self.nafnet = NAFNet(
            in_chans=out_chans,
            out_chans=out_chans,
        )
        
    def forward(self, res):
        res = self.unet(res)
        res = self.nafnet(res)
        return res
        