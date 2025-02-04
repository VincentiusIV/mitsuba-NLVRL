import os
import enoki as ek
import numpy as np
import mitsuba
mitsuba.set_variant('scalar_rgb')

from mitsuba.render import VolumeGrid

res = 16
albedo = np.zeros((res, res, res, 3))
X, Y, Z = np.meshgrid(np.linspace(0, 1,  res), np.linspace(0, 1,  res), np.linspace(0, 1,  res))
albedo[:, :, :, 2] = np.clip(0.5 * (np.sin(4 * np.pi * X + np.pi * np.cos(Y)) + 1), 0, 1)
albedo[:, :, :, 0] = 1 - albedo[:, :, :, 2]
albedo[:, :, :, 1] = 0.05
os.makedirs('textures', exist_ok=True)
VolumeGrid(albedo.astype(np.float32)).write('textures/albedo.vol')

# write_binary_grid3d('textures/albedo.vol', albedo)
