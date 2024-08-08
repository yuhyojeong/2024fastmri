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
        self.grappa_examples = []
        
        image_files = list(Path(root / "image").iterdir())
        kspace_files = list(Path(root / "kspace").iterdir())
        
        if not forward:
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
        
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]
            
        for fname in sorted(image_files):
            with h5py.File(fname, "r") as hf:
                num_slices = hf['image_grappa'].shape[0]
                self.grappa_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
##########
        self.image_examples = self.image_examples[:len(self.image_examples)]
        self.kspace_examples = self.kspace_examples[:len(self.kspace_examples)]
        self.grappa_examples = self.grappa_examples[:len(self.grappa_examples)]
        
    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
#         return 64
        return len(self.kspace_examples)
    
    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        grappa_fname, dataslice = self.grappa_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        with h5py.File(grappa_fname, "r") as hf:
            grappa = hf['image_grappa'][dataslice]
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, grappa, target, attrs, kspace_fname.name, dataslice)


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
#         pin_memory = True,
#         num_workers = 2
    )
    return data_loader

