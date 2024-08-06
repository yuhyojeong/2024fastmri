import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, augmentor = None): ##
        self.isforward = isforward
        self.max_key = max_key
        ##
        self.augmentor = augmentor
    def __call__(self, mask, input, grappa, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
            
        grappa = to_tensor(grappa)
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        ###
        if not self.isforward:
            if self.augmentor is not None:
                if self.augmentor.schedule_p() > 0.0:                
                    kspace, _ = self.augmentor(kspace, target.shape)
                    if (kspace.shape[-2] == 768):
                        kspace = kspace.permute(0, 2, 1, 3)
        ###

        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, grappa, target, maximum, fname, slice
