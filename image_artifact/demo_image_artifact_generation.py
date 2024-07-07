import numpy as np

import matplotlib.pyplot as plt

img = plt.imread('lenna.png')

# gray image generation
# img = np.mean(img, axis=2, keepdims=True)

sz = img.shape
cmap = 'gray' if sz[2] == 1 else None

# plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
# plt.title('Ground Truth')
# plt.show()

## 1-1. Inpainting: Uniform Sampling
# ds_y = 2
# ds_x = 4

# msk = np.zeros(sz)
# msk[::ds_y, ::ds_x, :] = 1

# dst = img * msk
# plt.subplot(131)
# plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
# plt.title('Ground Truth')

# plt.subplot(132)
# plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
# plt.title('Uniform Sampling Mask')

# plt.subplot(133)
# plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
# plt.title('Uniform Sampling')

# plt.show()

## 1-2. Inpainting: Random Sampling

# 완전 랜덤 샘플링
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# prob = 0.5
# msk = (rnd > prob).astype(np.float32)

# # 채널 방향으로는 동일한 샘플링하기
# rnd = np.random.rand(sz[0], sz[1], 1)
# prob = 0.5
# msk = (rnd > prob).astype(np.float32)
# msk = np.tile(msk, [1, 1, sz[2]])

# dst = img*msk

# plt.subplot(131)
# plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
# plt.title('Ground Truth')

# plt.subplot(132)
# plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
# plt.title('Random Sampling Mask')

# plt.subplot(133)
# plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
# plt.title('Random Sampling')

# plt.show()

## 1-3. Inpainting: Gaussian Sampling
ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])

x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1
a = 1

# 모든 채널에 대해 랜덤
# gaus = a * np.exp(-((x-x0)**2/(2*sgmx**2) + (y-y0)**2/(2*sgmy**2)))
# gaus = np.tile(gaus[:,:,np.newaxis], (1,1,sz[2]))
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# msk = (rnd < gaus).astype(np.float32)

# 채널 방향으로 동일한 샘플링을 가지도록하기
gaus = a * np.exp(-((x-x0)**2/(2*sgmx**2) + (y-y0)**2/(2*sgmy**2)))
gaus = np.tile(gaus[:,:,np.newaxis], (1,1,1))
rnd = np.random.rand(sz[0], sz[1], 1)
msk = (rnd < gaus).astype(np.float32)
msk = np.tile(msk, (1, 1, sz[2]))

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('Ground Truth')

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title('Gaussian Sampling Mask')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('Gaussian Sampling')

plt.show()
