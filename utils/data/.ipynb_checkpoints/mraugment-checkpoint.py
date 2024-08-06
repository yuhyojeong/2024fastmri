import numpy as np
from math import exp
import torch
import torchvision.transforms.functional as TF
from fastmri.data import transforms as T
from fastmri import fft2c, ifft2c, rss_complex, complex_abs

def complex_channel_first(x):
    assert x.shape[-1] == 2
    if len(x.shape) == 3:
        # Single-coil (H, W, 2) -> (2, H, W)
        x = x.permute(2, 0, 1)
    else:
        # Multi-coil (C, H, W, 2) -> (2, C, H, W)
        assert len(x.shape) == 4
        x = x.permute(3, 0, 1, 2)
    return x

def complex_channel_last(x):
    assert x.shape[0] == 2
    if len(x.shape) == 3:
        # Single-coil (2, H, W) -> (H, W, 2)
        x = x.permute(1, 2, 0)
    else:
        # Multi-coil (2, C, H, W) -> (C, H, W, 2)
        assert len(x.shape) == 4
        x = x.permute(1, 2, 3, 0)
    return x


class AugmentationPipeline:
    """
    Describes the transformations applied to MRI data and handles
    augmentation probabilities including generating random parameters for 
    each augmentation.
    """
    def __init__(self, hparams):
        self.hparams = hparams
        self.weight_dict ={
                      'translation': hparams.aug_weight_translation,
                      'rotation': hparams.aug_weight_rotation,
                      'scaling': hparams.aug_weight_scaling,
                      'shearing': hparams.aug_weight_shearing,
                      'rot90': hparams.aug_weight_rot90,
                      'fliph': hparams.aug_weight_fliph,
                      'flipv': hparams.aug_weight_flipv
        }
        self.upsample_augment = hparams.aug_upsample
        self.upsample_factor = hparams.aug_upsample_factor
        self.upsample_order = hparams.aug_upsample_order
        self.transform_order = hparams.aug_interpolation_order
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def augment_image(self, im, max_output_size=None):
        # Trailing dims must be image height and width (for torchvision) 
        im = complex_channel_first(im)
        
        # ---------------------------  
        # pixel preserving transforms
        # ---------------------------  
        # Horizontal flip
        if self.random_apply('fliph'):
            im = TF.hflip(im)

        # Vertical flip 
        if self.random_apply('flipv'):
            im = TF.vflip(im)

        # Rotation by multiples of 90 deg 
        if self.random_apply('rot90'):
            k = self.rng.randint(1, 4)  
            im = torch.rot90(im, k, dims=[-2, -1])

        # Translation by integer number of pixels
        if self.random_apply('translation'):
            h, w = im.shape[-2:]
            t_x = self.rng.uniform(-self.hparams.aug_max_translation_x, self.hparams.aug_max_translation_x)
            t_x = int(t_x * h)
            t_y = self.rng.uniform(-self.hparams.aug_max_translation_y, self.hparams.aug_max_translation_y)
            t_y = int(t_y * w)
            
            pad, top, left = self._get_translate_padding_and_crop(im, (t_x, t_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.crop(im, top, left, h, w)

        # ------------------------       
        # interpolating transforms
        # ------------------------  
        interp = False 

        # Rotation
        if self.random_apply('rotation'):
            interp = True
            rot = self.rng.uniform(-self.hparams.aug_max_rotation, self.hparams.aug_max_rotation)
        else:
            rot = 0.

        # Shearing
        if self.random_apply('shearing'):
            interp = True
            shear_x = self.rng.uniform(-self.hparams.aug_max_shearing_x, self.hparams.aug_max_shearing_x)
            shear_y = self.rng.uniform(-self.hparams.aug_max_shearing_y, self.hparams.aug_max_shearing_y)
        else:
            shear_x, shear_y = 0., 0.

        # Scaling
        if self.random_apply('scaling'):
            interp = True
            scale = self.rng.uniform(1-self.hparams.aug_max_scaling, 1 + self.hparams.aug_max_scaling)
        else:
            scale = 1.

        # Upsample if needed
        upsample = interp and self.upsample_augment
        if upsample:
            upsampled_shape = [im.shape[-2] * self.upsample_factor, im.shape[-1] * self.upsample_factor]
            original_shape = im.shape[-2:]
            interpolation  = TF.InterpolationMode.BICUBIC if self.upsample_order == 3 else TF.InterpolationMode.BILINEAR
            im = TF.resize(im, size=upsampled_shape, interpolation=interpolation)

        # Apply interpolating transformations 
        # Affine transform - if any of the affine transforms is randomly picked
        if interp:
            h, w = im.shape[-2:]
            pad = self._get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.affine(im,
                           angle=rot,
                           scale=scale,
                           shear=(shear_x, shear_y),
                           translate=[0, 0],
                           interpolation=TF.InterpolationMode.BILINEAR
                          )
            im = TF.center_crop(im, (h, w))
        
        # ---------------------------------------------------------------------
        # Apply additional interpolating augmentations here before downsampling
        # ---------------------------------------------------------------------
        
        # Downsampling
        if upsample:
            im = TF.resize(im, size=original_shape, interpolation=interpolation)
        
        # Final cropping if augmented image is too large
        if max_output_size is not None:
            im = crop_if_needed(im, max_output_size)
            
        # Reset original channel ordering
        im = complex_channel_last(im)
        
        return im
    
    def augment_from_kspace(self, kspace, target_size, max_train_size=None):       
        im = ifft2c(kspace) 
        im = self.augment_image(im, max_output_size=max_train_size)
        target = self.im_to_target(im, target_size)
        kspace = fft2c(im)
        return kspace, target
    
    def im_to_target(self, im, target_size):     
        # Make sure target fits in the augmented image
        cropped_size = [min(im.shape[-3], target_size[0]), 
                        min(im.shape[-2], target_size[1])]
        
        if len(im.shape) == 3: 
            # Single-coil
            target = complex_abs(T.complex_center_crop(im, cropped_size))
        else:
            # Multi-coil
            assert len(im.shape) == 4
            target = T.center_crop(rss_complex(im), cropped_size)
        return target  
            
    def random_apply(self, transform_name):
        if self.rng.uniform() < self.weight_dict[transform_name] * self.augmentation_strength:
            return True
        else: 
            return False
        
    def set_augmentation_strength(self, p):
        self.augmentation_strength = p

    @staticmethod
    def _get_affine_padding_size(im, angle, scale, shear):
        """
        Calculates the necessary padding size before applying the 
        general affine transformation. The output image size is determined based on the 
        input image size and the affine transformation matrix.
        """
        h, w = im.shape[-2:]
        corners = [
            [-h/2, -w/2, 1.],
            [-h/2, w/2, 1.], 
            [h/2, w/2, 1.], 
            [h/2, -w/2, 1.]
        ]
        mx = torch.tensor(TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2,3)
        corners = torch.cat([torch.tensor(c).reshape(3,1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h-1) 
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2),  min=0.0, max=w-1)
        return int(py.item()), int(px.item())

    @staticmethod
    def _get_translate_padding_and_crop(im, translation):
        t_x, t_y = translation
        h, w = im.shape[-2:]
        pad = [0, 0, 0, 0]
        if t_x >= 0:
            pad[3] = min(t_x, h - 1) # pad bottom
            top = pad[3]
        else:
            pad[1] = min(-t_x, h - 1) # pad top
            top = 0
        if t_y >= 0:
            pad[0] = min(t_y, w - 1) # pad left
            left = 0
        else:
            pad[2] = min(-t_y, w - 1) # pad right
            left = pad[2]
        return pad, top, left

class DataAugmentor:
    """
    High-level class encompassing the augmentation pipeline and augmentation
    probability scheduling. A DataAugmentor instance can be initialized in the 
    main training code and passed to the DataTransform to be applied 
    to the training data.
    """
        
    def __init__(self, hparams, current_epoch_fn):
        """
        hparams: refer to the arguments below in add_augmentation_specific_args
        current_epoch_fn: this function has to return the current epoch as an integer 
        and is used to schedule the augmentation probability.
        """
        self.current_epoch_fn = current_epoch_fn
        self.hparams = hparams
        self.augmentation_pipeline = AugmentationPipeline(hparams)
        
    def __call__(self, kspace, target_size):
        """
        Generates augmented kspace and corresponding augmented target pair.
        kspace: torch tensor of shape [C, H, W, 2] (multi-coil) or [H, W, 2]
            where last dim is for real/imaginary channels
        target_size: [H, W] shape of the generated augmented target
        """
        # Set augmentation probability
        p = self.schedule_p()
        self.augmentation_pipeline.set_augmentation_strength(p)
        
        # Augment if needed
        if p > 0.0:
            kspace, target = self.augmentation_pipeline.augment_from_kspace(kspace,
                                                                          target_size=target_size)
                    
        return kspace, target
        
    def schedule_p(self):
        D = self.hparams.aug_delay
        T = self.hparams.num_epochs
        t = self.current_epoch_fn
        p_max = self.hparams.aug_strength
        if t <= D:
            return 0.0
        else:
            if self.hparams.aug_schedule == 'constant':
                p = p_max
            elif self.hparams.aug_schedule == 'ramp':
                p = (t-D)/(T-D) * p_max
            elif self.hparams.aug_schedule == 'exp':
                c = self.hparams.aug_exp_decay/(T-D) # Decay coefficient
                p = p_max/(1-exp(-(T-D)*c))*(1-exp(-(t-D)*c))
            return p