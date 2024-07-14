import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]
        random.shuffle(self.image_examples)
        random.shuffle(self.kspace_examples)
        self.image_examples = self.image_examples[:len(self.image_examples) // 2]
        self.kspace_examples = self.kspace_examples[:len(self.kspace_examples) // 2]
        
    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])

        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        # Return padded or resized data
        input = pad_data(input)  # Adjust according to your dataset
        mask = pad_mask(mask)
#         if not self.forward:
#             target = self.pad_data(target, self.max_target_shape)
        
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)

    
def pad_data(arr):
    """
    Pad the input tensor to have max_channels in the channel dimension.

    """
    channels, _, width = arr.shape
    pad_w = 396 - width
    pad_c = 20-channels
    padding = ((0, pad_c), (0, 0), (0, pad_w))  # (2, 2, width, width, height, height, channel, channel, slice, slice)
    arr = np.pad(arr, padding)
    return arr

def pad_mask(arr):
    pad = 396 - arr.shape[0]
    mask = np.pad(arr, (0, pad))
    return mask

def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
