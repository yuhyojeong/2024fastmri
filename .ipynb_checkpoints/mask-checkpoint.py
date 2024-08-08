import h5py
import numpy as np

file11= '/home/Data/train/kspace/brain_acc4_1.h5'
file22 = '/home/Data/train/kspace/brain_acc5_1.h5'
file33 = '/home/Data/train/kspace/brain_acc8_1.h5'
file44 = '/home/Data/leaderboard/acc9/kspace/brain_test1.h5'
file1 = '/home/Data/train/kspace/brain_acc4_111.h5'
file2 = '/home/Data/train/kspace/brain_acc5_70.h5'
file3 = '/home/Data/train/kspace/brain_acc8_21.h5'


def cf(mask):
    total_points = mask.shape[0]
    center = total_points // 2
    start = center
    end = center
    while start > 0 and mask[start] == 1:
        start -= 1
    while end < total_points and mask[end] == 1:
        end += 1

    fully_sampled_points = end - start - 1
    center_fraction = fully_sampled_points / total_points
    
    return center_fraction

# Open the .h5 file
with h5py.File(file11, 'r') as h5_file:
    mask = np.array(h5_file['mask'])
    print("x4", mask.shape, cf(mask))
with h5py.File(file22, 'r') as h5_file:
    mask = np.array(h5_file['mask'])
    print("x5", mask.shape, cf(mask))
with h5py.File(file33, 'r') as h5_file:
    mask = np.array(h5_file['mask'])
    print("x8", mask.shape, cf(mask))
with h5py.File(file44, 'r') as h5_file:
    mask = np.array(h5_file['mask'])
    print("x9", mask.shape, cf(mask))
    
    
with h5py.File(file1, 'r') as h5_file:
    mask = np.array(h5_file['mask'])
    print("x4", mask.shape, cf(mask))
with h5py.File(file2, 'r') as h5_file:
    mask = np.array(h5_file['mask'])
    print("x5", mask.shape, cf(mask))
with h5py.File(file3, 'r') as h5_file:
    mask = np.array(h5_file['mask'])
    print("x8", mask.shape, cf(mask))

