from core.utils import _get_uncertain_pt_coords_randomness, get_uncertain_point_coords_on_grid, _uncertainty, \
    _get_uncertain_point_coords_on_grid
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
array = np.load('test.npy')[0]
print(array.shape)
# array = cv2.resize(array, dsize=(8, 8))
# array = cv2.resize(array, dsize=(512, 512))
print(np.max(array))
print(np.min(array))

# array[100:412, 100:412, ...] = 0.5
# array[100:412, 100:412, ...] = 0.5
array = array[None, :, :]
# plt.imshow(array)
# plt.show()
# np.reshape()
uncertainty_map = _uncertainty(array, cls=0)
_, coords = _get_uncertain_point_coords_on_grid(uncertainty_map, rate=0.03)
coords = coords[..., ::-1]
# print(coords)

print(np.max(coords))
coords = coords.numpy()
coords = coords * 512
coords = coords.astype(np.int)
# coords = np.squeeze(coords)
print(coords.shape)
array = np.squeeze(array)
ori_array = array.copy()
array = np.zeros_like(array)
array[array == 0.5] = 0.3

# array = (array-np.min(array))/(np.max(array)-np.min(array))
array[coords[..., 0], coords[..., 1]] = 1 - array[coords[..., 0], coords[..., 1]]
# array[coords[..., 0], coords[..., 1]][array[coords[..., 0], coords[..., 1]] == 0.5] = 1
array = np.concatenate([array[..., None]] * 3, axis=-1)
# np.save('test.npy',array)
# array = np.load('test.npy',allow_pickle=True)
print(array)
ori_array = np.concatenate([ori_array[..., None]] * 3, axis=-1)
array = np.concatenate([array, ori_array], axis=-2)
# array = array
plt.imshow(array)
plt.show()
plt.close()
