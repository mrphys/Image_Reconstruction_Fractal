# Standard imports
import os
import glob
import json
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import h5py
from scipy.ndimage import zoom
import tensorflow_mri as tfmr
import tensorflow as tf
import mapvbvd
import shutil
import sys
sys.path.append('inter4k_utils')
import preprocessing_multicoil_noselect as preproc_multicoil
import preprocessing_natural_videos as preprocessing_natural_videos
# import preprocessing_natural_videos_flow as preproc_flow
import preprocessing_trajectory_gen as preproc_traj
# import preprocessing_fastdvdnet_noselect as preproc_fastdvdnet
# import preprocessing_cartesian_varnet as preproc_varnet
# from subsample_cartesian import generate_mask
# import recon_utils
# import create_dataset as data_util
# import neptune.new as neptune
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from skimage.transform import resize as sc_resize
from skimage.exposure import equalize_adapthist
from scipy.ndimage import median_filter
import h5py
import random
import zlib
from collections import defaultdict
from time import perf_counter
#import utils
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def display_gifs(img,out_path, patient):
    # Plot the results and create an animation
    fig, axs = plt.subplots(1, figsize=(6, 3))
    plt.title(f'Image_shape: {img.shape}')
    frames = []

    # Create frames for the animation
    for i in range(img.shape[0]):
        p1 = axs.imshow(np.squeeze(img)[i,...], cmap='gray',vmin = np.min(img), vmax=np.max(img))
        text = axs.text(0.05, 0.95, f'Frame {i+1}', color='white', fontsize=12, ha='left', va='top', transform=axs.transAxes)
        frames.append([p1,text])  # Append the current frames to the list

    # Create and save animation
    ani = animation.ArtistAnimation(fig, frames)
    Path(f'{out_path}').mkdir(parents=True, exist_ok=True)
    ani.save(f'{out_path}/{patient}.gif', fps=10)


import enum
RNG = tf.random.Generator.from_seed(8745)

class CartDims(enum.IntEnum):
  COL = -1
  LIN = -2
  CHA = -3
  PHS = -4
  SLC = -5
  
class NonCartDims(enum.IntEnum):
  COL = -1
  CHA = -2
  PHS = -3
  SLC = -4
  
class Transformer():
  def __init__(self):
    pass
  def fit(self, x):
    raise NotImplementedError()
  def transform(self, x):
    raise NotImplementedError()
  def fit_transform(self, x):
    return self.fit(x).transform(x)
  
  
class LowPassScaler(Transformer):
  
  def __init__(self, cutoff=0.5 * np.pi):
    self.cutoff = cutoff
    self.scale = None
    
  def fit(self, kspace, trajectory, density):
    kspace = tfmr.signal.filter_kspace(
        kspace, trajectory=trajectory)
    image = tfmr.recon.adjoint(
        kspace, image_shape=[256, 256], trajectory=trajectory, density=density)
    # Collapse all frames.
    image = tf.math.reduce_mean(image, axis=-3)
    maxval = tf.math.reduce_max(tf.math.abs(image))
    self.scale = tf.math.reciprocal(maxval)
    return self
  
  def transform(self, kspace):
    if self.scale is None:
      raise ValueError(
          f"{self.__class__.__name__} has not been fitted yet. "
          f"You need to call `fit` before calling `transform`.")
    return kspace * tf.cast(self.scale, kspace.dtype)
  
def _scale_kspace_data(kspace, trajectory, density):
    # This value is selected so that reconstruction renders images approximately
    # between 0 and 1.
    # scaler = LogScaler(log_mean=-4.424523).fit(kspace)
    # scaler = EmpiricalScaler(scale=1000.).fit(kspace)
    scaler = LowPassScaler().fit(kspace, trajectory, density)
    return scaler.transform(kspace)
  
def _apply_coil_engineering(kspace):
    kspace = tfmr.coils.compress_coils(kspace, coil_axis=-3, out_coils=12)
    return kspace
  
def _compute_trajectory_and_density(image_shape, extra_shape,views):
    """Returns the trajectory and sampling density."""
    base_resolution = image_shape[-1]
    # views = 10 #10-30
    phases = extra_shape[-1]
    # Compute trajectory.
    trajectory = tfmr.sampling.radial_trajectory(base_resolution=base_resolution,
                                                views=views,
                                                phases=phases,
                                                ordering='sorted_half')
    # Compute density.
    density = tfmr.sampling.estimate_radial_density(trajectory)
    # Flatten trajectory and density.
    trajectory = tfmr.sampling.flatten_trajectory(trajectory)
    density = tfmr.sampling.flatten_density(density)
    return trajectory, density

def _compute_trajectory_and_density2(image_shape, extra_shape,views):
    """Returns the trajectory and sampling density."""
    base_resolution = image_shape[-1]
    # views = 10 #10-30
    phases = extra_shape[-1]
    # Compute trajectory.
    trajectory = tfmr.sampling.radial_trajectory(base_resolution=base_resolution,
                                                views=views,
                                                phases=phases,
                                                ordering='tiny_half')
    # Compute density.
    density = tfmr.sampling.estimate_radial_density(trajectory)
    # Flatten trajectory and density.
    trajectory = tfmr.sampling.flatten_trajectory(trajectory)
    density = tfmr.sampling.flatten_density(density)
    return trajectory, density

def compute_trajectory_and_density(image_shape,extra_shape,views,ordering):
    """Returns the trajectory and sampling density."""
    base_resolution = image_shape[-1]
    phases = extra_shape[-1]
    trajectory = tfmr.sampling.radial_trajectory(base_resolution=base_resolution,
                                                views=views,
                                                phases=phases,
                                                ordering=ordering)
    # Compute density.
    density = tfmr.sampling.estimate_radial_density(trajectory)
    # Flatten trajectory and density.
    trajectory = tfmr.sampling.flatten_trajectory(trajectory)
    density = tfmr.sampling.flatten_density(density)
    return trajectory, density
  
def _make_fs_trigcine_image(kspace, image_shape):
  """Returns a fully sampled image from k-space."""
  # Crop to fixed size.
  image = tfmr.signal.ifft(kspace, axes=[-2, -1], shift=True)
  image = tfmr.resize_with_crop_or_pad(image, image_shape)
  # Random 90-degree rotation.
  shape = tf.shape(image)
  image = tf.reshape(image, [-1, shape[-2], shape[-1], 1])  # [batch, rows, cols, 1]
  image = tf.image.rot90(image, k=RNG.uniform([], 0, 4, dtype=tf.int32))
  image = tf.reshape(image, shape)
  # Get number of phases in image.
  num_phases = tf.shape(image)[CartDims.PHS]
  extra_shape = [num_phases]
  # This is needed to avoid errors with static shape later.
  extra_shape = tf.convert_to_tensor(extra_shape)
  extra_shape = tf.ensure_shape(extra_shape, (1,))
  # Scale image to unit max.
  # scale = tf.math.reduce_max(tf.math.abs(image))
  # image = image / tf.cast(scale, image.dtype)
  return image, extra_shape

def _sample_noncartesian_kspace(image, trajectory):
  """Samples a non-Cartesian k-space from an image."""
  kspace = tfmr.signal.nufft(image, trajectory,
                              transform_type='type_2',
                              fft_direction='forward')
  image_shape = tf.shape(image)[-2:]
  kspace /= tf.math.sqrt(
      tf.cast(tf.math.reduce_prod(image_shape), tf.complex64))
  return kspace

def recon_adjoint(kspace, operator):
  kspace = tf.convert_to_tensor(kspace)
  kspace = operator.preprocess(kspace, adjoint=True)
  image = operator.transform(kspace, adjoint=True)
  image = operator.postprocess(image, adjoint=True)
  return image

def convert_to_real(complex_tensor):
  # Get the real and imaginary parts
  real_part = tf.math.real(complex_tensor)
  imag_part = tf.math.imag(complex_tensor)
  stacked = tf.concat([real_part, imag_part], axis=0)
  return stacked

def convert_to_complex(stacked_tensor):
    real_part, imag_part = tf.split(stacked_tensor, 2, axis=-1)
    complex_tensor = tf.complex(real_part, imag_part)
    return complex_tensor

def scale_frames(img):
  required_frames = 20 
  zooms = np.ones(img.ndim)
  zooms[0] = 20/img.shape[0]
  img = zoom(img, zooms, order=0)
  return img

def circular_pad(img, padding_fraction=0.3):
    num_frames = img.shape[0]
    padding_size = int(num_frames * padding_fraction)

    # Wrap frames from both ends for padding
    padded_img = np.concatenate([img[-padding_size:], img, img[:padding_size]], axis=0)
    
    return padded_img

def crop_or_circular_pad(img, output_frames):
   if img.shape[0] >= output_frames:
      return img[0:output_frames]
   else:
      padding_size = output_frames - img.shape[0]
      out_img = img
      while(True):
         if padding_size >= img.shape[0]:
            out_img = np.concatenate([img, out_img], axis = 0)
            padding_size = padding_size - img.shape[0]
         elif padding_size == 0:
            break
         else:
            out_img = np.concatenate([out_img, img[:padding_size]], axis = 0)
            break
      return out_img
            

def normalise(image):
    scale = tf.math.reduce_max(tf.math.abs(image))
    image = image / tf.cast(scale, image.dtype)
    return image

#check for zero frames and circular crop/pad before processing data
def process_kspace(kspace):
    """Preprocesses a Wip1200 input.
    inputs should always have a k-space tensor, although its content differs
    depending on the context
        - if `training=True`, `kspace` is a fully-sampled Cartesian *k*-space
        with shape `(phases, coils, rows, cols)`.
        - if `training=False`, `kspace` is an undersampled radial *k*-space.
    
    1) cast as float
    2) make fs trigcine image
      -get image via inverse fft
      -perform augmentation (90rot)
      -make sure shape is correct
      -scale image by dividing by max val
      -find number of phases and save to extra_shape
    3)compute trajectory and density
    4) rearrange img matrix to channels,time,height,width and kspace to channels,time,space(CHECK dims)
    5)expand dims of trajectory and density
    6)sample non-cartesian kspace using image
    7)compress coils
    8)get adjoint to convert back to image space
    9)get ground truth image from (2)
    10)combine coils of ground truth image , take abs and expand dims
    """
    # This is not used for prospective data
    kspace = tf.cast(kspace, dtype = tf.complex64)
    phase_mask = tf.math.reduce_all(tf.abs(kspace) == 0.0, axis=(-2, -1))
    #kspace = kspace[~phase_mask]
    kspace = tf.ragged.boolean_mask(kspace, ~phase_mask).to_tensor()
    kspace = crop_or_circular_pad(kspace, 24)
    image_shape = tf.constant([256, 256], dtype=tf.int32)
    image, extra_shape = _make_fs_trigcine_image(kspace, image_shape)
    image = tf.einsum('...tchw->...cthw', image) #check dims
    #kspace = tf.einsum('...tcs->...cts', kspace) #check dims
    trajectory,density = _compute_trajectory_and_density(image_shape, extra_shape)
    trajectory = tf.expand_dims(trajectory, axis=-4)
    density = tf.expand_dims(density, axis=-3)
    kspace = _sample_noncartesian_kspace(image, trajectory)
    kspace = tfmr.coils.compress_coils(kspace, coil_axis=-3, out_coils=12)
    operator = tfmr.linalg.LinearOperatorMRI(
      image_shape=image_shape,
      extra_shape=extra_shape,
      trajectory=trajectory,
      density=density)
    undersampled_image = recon_adjoint(kspace, operator)
    
    #undersampled_image  = tfmr.coils.combine_coils(undersampled_image, coil_axis=-4)
    undersampled_image = normalise(undersampled_image)
    undersampled_image = convert_to_real(undersampled_image)#USE FOR MULTICOIL
    undersampled_image = np.moveaxis(undersampled_image, [1,2,3,0], [0,1,2,3])
    #undersampled_image = tf.math.abs(undersampled_image)#USE FOR SINGLE COIL
    target_image = tfmr.coils.combine_coils(image, coil_axis=-4)
    target_image = normalise(target_image)
    target_image = tf.math.abs(target_image)
    # undersampled_image = scale_frames(undersampled_image)
    # target_image = scale_frames(target_image)
    # undersampled_image = circular_pad(undersampled_image)
    # target_image = circular_pad(target_image)
    return (undersampled_image, target_image)

def resize(image,zoom_factors):
    #zoom_factors = (zoom_val,zoom_val,zoom_val)
    resized_image = zoom(image, zoom_factors, order=0)
    return resized_image

def process_data(raw_data_path,out_path):
    patients = np.unique([pat.split('/')[-1].split('.')[0] for pat in glob.glob(f'{raw_data_path}/*.h5')])
    print(patients)
    for i,patient in enumerate(patients):
        print(i)
        try:
          file_path = f'{raw_data_path}/{patient}.h5'
          kspace = np.squeeze(np.array(h5py.File(file_path, 'r')['kspace']))
          if np.min(kspace.shape)<12 or kspace.ndim !=4:
              print('skipping due to invalid input...')
              continue
          print(kspace.shape, np.max(kspace), np.min(kspace))
          x = process_kspace(np.squeeze(kspace))
          Path(out_path).mkdir(parents=True, exist_ok=True)
          x,y = process_kspace(np.squeeze(kspace))
          Path(out_path).mkdir(parents=True, exist_ok=True)
          np.save(f'{out_path}/{patient}_output', y)
          np.save(f'{out_path}/{patient}_input', x)
        except Exception as e:
          print(f'Skipping {patient} because of Exception: {e}')

#check for zero frames and circular crop/pad before processing data
def process_superres_kspace(kspace):
    """Preprocesses a Wip1200 input.
    inputs should always have a k-space tensor, although its content differs
    depending on the context
        - if `training=True`, `kspace` is a fully-sampled Cartesian *k*-space
        with shape `(phases, coils, rows, cols)`.
        - if `training=False`, `kspace` is an undersampled radial *k*-space.
    
    1) cast as float
    2) make fs trigcine image
      -get image via inverse fft
      -perform augmentation (90rot)
      -make sure shape is correct
      -scale image by dividing by max val
      -find number of phases and save to extra_shape
    3)compute trajectory and density
    4) rearrange img matrix to channels,time,height,width and kspace to channels,time,space(CHECK dims)
    5)expand dims of trajectory and density
    6)sample non-cartesian kspace using image
    7)compress coils
    8)get adjoint to convert back to image space
    9)get ground truth image from (2)
    10)combine coils of ground truth image , take abs and expand dims
    """
    # This is not used for prospective data
    kspace = tf.cast(kspace, dtype = tf.complex64)
    phase_mask = tf.math.reduce_all(tf.abs(kspace) == 0.0, axis=(-2, -1))
    #kspace = kspace[~phase_mask]
    kspace = tf.ragged.boolean_mask(kspace, ~phase_mask).to_tensor()
    kspace = crop_or_circular_pad(kspace, 48)
    image_shape = tf.constant([256, 256], dtype=tf.int32)
    image, extra_shape = _make_fs_trigcine_image(kspace, image_shape)
    trajectory,density = _compute_trajectory_and_density(image_shape, extra_shape/2)
    undersampled_image = tf.reshape(image, (24, 2, *image.shape[1:]))
    undersampled_image = tf.reduce_mean(undersampled_image, axis=1) 
    image = tf.einsum('...tchw->...cthw', image) #check dims
    undersampled_image = tf.einsum('...tchw->...cthw', undersampled_image)
    #kspace = tf.einsum('...tcs->...cts', kspace) #check dims
    trajectory = tf.expand_dims(trajectory, axis=-4)
    density = tf.expand_dims(density, axis=-3)
    kspace = _sample_noncartesian_kspace(undersampled_image, trajectory)
    kspace = tfmr.coils.compress_coils(kspace, coil_axis=-3, out_coils=12)
    operator = tfmr.linalg.LinearOperatorMRI(
      image_shape=image_shape,
      extra_shape=tf.cast(extra_shape/2,tf.int32),
      trajectory=trajectory,
      density=density)
    undersampled_image = recon_adjoint(kspace, operator)
    
    #undersampled_image  = tfmr.coils.combine_coils(undersampled_image, coil_axis=-4)
    undersampled_image = normalise(undersampled_image)
    undersampled_image = convert_to_real(undersampled_image)#USE FOR MULTICOIL
    undersampled_image = np.moveaxis(undersampled_image, [1,2,3,0], [0,1,2,3])
    #undersampled_image = tf.math.abs(undersampled_image)#USE FOR SINGLE COIL
    target_image = tfmr.coils.combine_coils(image, coil_axis=-4)
    target_image = normalise(target_image)
    target_image = tf.math.abs(target_image)
    # undersampled_image = scale_frames(undersampled_image)
    # target_image = scale_frames(target_image)
    # undersampled_image = circular_pad(undersampled_image)
    # target_image = circular_pad(target_image)
    return (undersampled_image, target_image)


def process_superres_data(raw_data_path,out_path):
    patients = np.unique([pat.split('/')[-1].split('.')[0] for pat in glob.glob(f'{raw_data_path}/*.h5')])
    print(patients)
    for i,patient in enumerate(patients):
        print(i)
        try:
          file_path = f'{raw_data_path}/{patient}.h5'
          kspace = np.squeeze(np.array(h5py.File(file_path, 'r')['kspace']))
          if np.min(kspace.shape)<12 or kspace.ndim !=4:
              print('skipping due to invalid input...')
              continue
          print(kspace.shape, np.max(kspace), np.min(kspace))
          x,y = process_superres_kspace(np.squeeze(kspace))
          Path(out_path).mkdir(parents=True, exist_ok=True)
          np.save(f'{out_path}/{patient}_output', y)
          np.save(f'{out_path}/{patient}_input', x)
        except Exception as e:
          print(f'Skipping {patient} because of Exception: {e}')

def process_prospective_kspace2(kspace):
    """Preprocesses a Wip1200 input.
    inputs should always have a k-space tensor, although its content differs
    depending on the context
        - if `training=True`, `kspace` is a fully-sampled Cartesian *k*-space
        with shape `(phases, coils, rows, cols)`.
        - if `training=False`, `kspace` is an undersampled radial *k*-space.
    
    1) cast as float
    2) make fs trigcine image
      -get image via inverse fft
      -perform augmentation (90rot)
      -make sure shape is correct
      -scale image by dividing by max val
      -find number of phases and save to extra_shape
    3)compute trajectory and density
    4) rearrange img matrix to channels,time,height,width and kspace to channels,time,space(CHECK dims)
    5)expand dims of trajectory and density
    6)sample non-cartesian kspace using image
    7)compress coils
    8)get adjoint to convert back to image space
    9)get ground truth image from (2)
    10)combine coils of ground truth image , take abs and expand dims
    """
    # This is not used for prospective data
    kspace = tf.cast(kspace, dtype = tf.complex64)
    # image_shape = tf.constant([256, 256], dtype=tf.int32)
    # image, extra_shape = _make_fs_trigcine_image(kspace, image_shape)
    # trajectory,density = _compute_trajectory_and_density(image_shape, extra_shape)
    # image = tf.einsum('...tchw->...cthw', image) #check dims
    # #kspace = tf.einsum('...tcs->...cts', kspace) #check dims
    # trajectory = tf.expand_dims(trajectory, axis=-4)
    # density = tf.expand_dims(density, axis=-3)
    # #kspace = _sample_noncartesian_kspace(image, trajectory)
    '''(14, 28, 20, 13, 512)
    (14, 20, 6656)
    (1, 20, 6656, 2)
    (1, 20, 6656)
    (12, 20, 6656)'''
    #(512, 20, 13, 28, 14) --> (20,28,14,13,512) | (512,20,13,14)-->(20,14,13,512)
    print(kspace.shape)
    kspace =  tf.transpose(kspace, perm=[1,3,2,0]) #(coils, slices, time, spokes, readout)
    sh = kspace.shape
    print(kspace.shape)
    kspace = tf.reshape(kspace,(sh[0],sh[1],sh[2]*sh[3]))
    #kspace = kspace[:,sh[1]//2,...]
    print(kspace.shape)
    image_shape = tf.constant([256, 256], dtype=tf.int32)

    extra_shape = [kspace.shape[-2]]
    trajectory,density = _compute_trajectory_and_density2(image_shape, extra_shape,views=100, ordering='golden')
    trajectory = tf.expand_dims(trajectory, axis=-4)
    density = tf.expand_dims(density, axis=-3)
    kspace = tfmr.coils.compress_coils(kspace, coil_axis=-3, out_coils=12)
    print(kspace.shape)
    operator = tfmr.linalg.LinearOperatorMRI(
      image_shape=image_shape,
      extra_shape=extra_shape,
      trajectory=trajectory,
      density=density)
    undersampled_image = recon_adjoint(kspace, operator)
    #
    #undersampled_image  = tfmr.coils.combine_coils(undersampled_image, coil_axis=-4)
    undersampled_image = normalise(undersampled_image)
    undersampled_image = convert_to_real(undersampled_image)#USE FOR MULTICOIL
    undersampled_image = np.moveaxis(undersampled_image, [1,2,3,0], [0,1,2,3])
    #undersampled_image = tf.math.abs(undersampled_image)#USE FOR SINGLE COIL
    #undersampled_image = scale_frames(undersampled_image)
    #undersampled_image = circular_pad(undersampled_image)
    
    return undersampled_image



def process_prospective_kspace(kspace):
    """Preprocesses a Wip1200 input.
    inputs should always have a k-space tensor, although its content differs
    depending on the context
        - if `training=True`, `kspace` is a fully-sampled Cartesian *k*-space
        with shape `(phases, coils, rows, cols)`.
        - if `training=False`, `kspace` is an undersampled radial *k*-space.
    
    1) cast as float
    2) make fs trigcine image
      -get image via inverse fft
      -perform augmentation (90rot)
      -make sure shape is correct
      -scale image by dividing by max val
      -find number of phases and save to extra_shape
    3)compute trajectory and density
    4) rearrange img matrix to channels,time,height,width and kspace to channels,time,space(CHECK dims)
    5)expand dims of trajectory and density
    6)sample non-cartesian kspace using image
    7)compress coils
    8)get adjoint to convert back to image space
    9)get ground truth image from (2)
    10)combine coils of ground truth image , take abs and expand dims
    """
    # This is not used for prospective data
    kspace = tf.cast(kspace, dtype = tf.complex64)
    # image_shape = tf.constant([256, 256], dtype=tf.int32)
    # image, extra_shape = _make_fs_trigcine_image(kspace, image_shape)
    # trajectory,density = _compute_trajectory_and_density(image_shape, extra_shape)
    # image = tf.einsum('...tchw->...cthw', image) #check dims
    # #kspace = tf.einsum('...tcs->...cts', kspace) #check dims
    # trajectory = tf.expand_dims(trajectory, axis=-4)
    # density = tf.expand_dims(density, axis=-3)
    # #kspace = _sample_noncartesian_kspace(image, trajectory)
    '''(14, 28, 20, 13, 512)
    (14, 20, 6656)
    (1, 20, 6656, 2)
    (1, 20, 6656)
    (12, 20, 6656)'''
    #(512, 20, 13, 28, 14) --> (20,28,14,13,512) | (512,20,13,14)-->(20,14,13,512)
    print(kspace.shape)
    kspace =  tf.transpose(kspace, perm=[1,3,2,0]) #(coils, slices, time, spokes, readout)
    sh = kspace.shape
    print(kspace.shape)
    kspace = tf.reshape(kspace,(sh[0],sh[1],sh[2]*sh[3]))
    #kspace = kspace[:,sh[1]//2,...]
    print(kspace.shape)
    image_shape = tf.constant([256, 256], dtype=tf.int32)

    extra_shape = [kspace.shape[-2]]
    trajectory,density = _compute_trajectory_and_density2(image_shape, extra_shape,views=13)
    trajectory = tf.expand_dims(trajectory, axis=-4)
    density = tf.expand_dims(density, axis=-3)
    kspace = tfmr.coils.compress_coils(kspace, coil_axis=-3, out_coils=10)
    print(kspace.shape)
    operator = tfmr.linalg.LinearOperatorMRI(
      image_shape=image_shape,
      extra_shape=extra_shape,
      trajectory=trajectory,
      density=density)
    undersampled_image = recon_adjoint(kspace, operator)
    #
    #undersampled_image  = tfmr.coils.combine_coils(undersampled_image, coil_axis=-4)
    undersampled_image = normalise(undersampled_image)
    undersampled_image = convert_to_real(undersampled_image)#USE FOR MULTICOIL
    undersampled_image = np.moveaxis(undersampled_image, [1,2,3,0], [0,1,2,3])
    #undersampled_image = tf.math.abs(undersampled_image)#USE FOR SINGLE COIL
    #undersampled_image = scale_frames(undersampled_image)
    #undersampled_image = circular_pad(undersampled_image)
    
    return undersampled_image
    
def process_grasp_kspace(kspace):
    # (320, 18, 850, 69)
    kspace = tf.cast(kspace, dtype = tf.complex64)
    print(kspace.shape)
    kspace =  tf.transpose(kspace, perm=[1,3,2,0]) #(coils, slices, spokes, readout)
    kspace = kspace[...,:100]
    kspace = tf.pad(kspace, paddings=[[0,0],[16,0],[0,0],[0,0]], mode='constant', constant_values=0)
    sh = kspace.shape
    print(kspace.shape)
    kspace = tf.reshape(kspace,(sh[0],sh[1],sh[2]*sh[3]))
    #kspace = kspace[:,sh[1]//2,...]
    print(kspace.shape)
    image_shape = tf.constant([160, 160], dtype=tf.int32)
    image, extra_shape = _make_fs_trigcine_image(kspace, image_shape)
    target_image = tfmr.coils.combine_coils(image, coil_axis=-4)
    target_image = normalise(target_image)
    target_image = tf.math.abs(target_image)
    np.save()
    # undersampled_image = scale_frames(undersampled_image)
    # target_image = scale_frames(target_image)
    # undersampled_image = circular_pad(undersampled_image)
    # target_image = circular_pad(target_image)
    return target_image

    # trajectory,density = compute_trajectory_and_density(image_shape,views=13)
    # trajectory = tf.expand_dims(trajectory, axis=-4)
    # density = tf.expand_dims(density, axis=-3)
    # kspace = tfmr.coils.compress_coils(kspace, coil_axis=-3, out_coils=10)
    # print(kspace.shape)
    # operator = tfmr.linalg.LinearOperatorMRI(
    #   image_shape=image_shape,
    #   extra_shape=extra_shape,
    #   trajectory=trajectory,
    #   density=density)
    # undersampled_image = recon_adjoint(kspace, operator)
    # #
    # #undersampled_image  = tfmr.coils.combine_coils(undersampled_image, coil_axis=-4)
    # undersampled_image = normalise(undersampled_image)
    # undersampled_image = convert_to_real(undersampled_image)#USE FOR MULTICOIL
    # undersampled_image = np.moveaxis(undersampled_image, [1,2,3,0], [0,1,2,3])
    # #undersampled_image = tf.math.abs(undersampled_image)#USE FOR SINGLE COIL
    # #undersampled_image = scale_frames(undersampled_image)
    # #undersampled_image = circular_pad(undersampled_image)
    
    # return undersampled_image

def process_grasp_kspace_time_varying(kspace):
    # (320, 18, 850, 69)
    kspace = tf.cast(kspace, dtype = tf.complex64)
    print(kspace.shape)
    kspace =  tf.transpose(kspace, perm=[1,3,2,0]) #(coils, slices, spokes, readout) (18,69,850,320)
    # kspace = tf.pad(kspace, paddings=[[0,0],[16,0],[0,0],[0,0]], mode='constant', constant_values=0)
    # kspace = kspace[:,:52,:,:]
    sh = kspace.shape
    print(kspace.shape)
    kspace = tf.reshape(kspace,(sh[0],sh[1],sh[2]*sh[3]))
    #kspace = kspace[:,sh[1]//2,...]
    print(kspace.shape)
    image_shape = tf.constant([288, 288], dtype=tf.int32)
    extra_shape = [None]#tf.constant([1], dtype=tf.int32)
    kspace = tfmr.signal.ifft(kspace, axes=[-2], shift=True)

    trajectory,density = compute_trajectory_and_density(image_shape,extra_shape,views=sh[-2], ordering='golden_half')
    print(trajectory.shape, density.shape)
    trajectory = tf.expand_dims(trajectory, axis=0)
    density = tf.expand_dims(density, axis=0)
    # trajectory = tf.tile(trajectory, multiples=[1,sh[1], 1, 1])
    # density = tf.tile(density, multiples=[1,sh[1], 1])
    # extra_shape = tf.constant([sh[1]], dtype=tf.int32)
    # print(trajectory.shape, density.shape)
    # kspace = tfmr.coils.compress_coils(kspace, coil_axis=-3, out_coils=10)
    # print(kspace.shape)
    undersampled_image = []
    for j in range(sh[-1]):   # 
        undersampled_slice = []
        for i in range(0,sh[-2]-20,20):
            start = i*sh[3]
            end = (i+20)*sh[3]
            operator = tfmr.linalg.LinearOperatorMRI(
                            image_shape=image_shape,
                            extra_shape=None,
                            trajectory=trajectory[:,start:end],
                            density=density[:,start:end])
            undersampled_slice.append(recon_adjoint(kspace[:,j,start:end], operator))
        undersampled_slice= tf.stack(undersampled_slice, axis=1)
        print(f"SLICE SHAPE: {undersampled_slice.shape}")
        undersampled_image.append(undersampled_slice)
    undersampled_image = tf.stack(undersampled_image, axis=-1)
    print(f"IMAGE SHAPE: {undersampled_image.shape}")
    # undersampled_image  = tfmr.coils.combine_coils(undersampled_image, coil_axis=-4)
    undersampled_image = tfmr.coils.compress_coils(undersampled_image, coil_axis=0, out_coils=10)
    undersampled_image = normalise(undersampled_image)
    undersampled_image = convert_to_real(undersampled_image)#USE FOR MULTICOIL

    undersampled_image = np.moveaxis(undersampled_image, [1,2,3,4,0], [0,1,2,3,4])
    print(f"IMAGE SHAPE: {undersampled_image.shape}")
    # # undersampled_image = tf.math.abs(undersampled_image)#USE FOR SINGLE COIL
    # #undersampled_image = scale_frames(undersampled_image)
    # # #undersampled_image = circular_pad(undersampled_image)
    return undersampled_image


def process_grasp_data(raw_data_path, out_path):
    patients = np.unique([pat.split('/')[-1].split('.')[0] for pat in glob.glob(f'{raw_data_path}/*.npy')])
    for patient in patients:
        # twixObj =  mapvbvd.mapVBVD(f'{raw_data_path}/{patient}.dat')   
        # twixObj = twixObj[1]
        # twixObj.image.flagRemoveOS = False
        # raw_data = twixObj.image['']
        # print(patient, raw_data.shape)
        # # # raw_data = raw_data[:,:,:,0,:,0,:,0,0,0,0,0,0,0,0,0] 
        # raw_data = np.squeeze(raw_data)
        # # np.save(patient, raw_data)
        raw_data = np.squeeze(np.load(f'{raw_data_path}/{patient}.npy'))
        # if os.path.exists(out_path):
        #         shutil.rmtree(out_path)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        print(raw_data.shape)
        x= process_grasp_kspace_time_varying(raw_data)
        np.save(f'{out_path}/{patient}.npy', x)

# def process_grasp_data(raw_data_path, out_path):
#     patients = np.unique([pat.split('/')[-1].split('.')[0] for pat in glob.glob(f'{raw_data_path}/*.dat')])
#     for patient in patients:
#         twixObj =  mapvbvd.mapVBVD(f'{raw_data_path}/{patient}.dat')   
#         twixObj = twixObj[1]
#         twixObj.image.flagRemoveOS = False
#         raw_data = twixObj.image['']
#         # raw_data = raw_data[:,:,:,0,:,0,:,0,0,0,0,0,0,0,0,0] 
#         raw_data = np.squeeze(raw_data)
#         # if os.path.exists(out_path):
#         #         shutil.rmtree(out_path)
#         Path(out_path).mkdir(parents=True, exist_ok=True)
#         print(raw_data.shape)
#         x= process_grasp_kspace(raw_data)
#         np.save(f'{out_path}/{patient}.npy', x)

def convert_multicoil_to_image(multicoil):
    complex_tensor = convert_to_complex(multicoil)
    complex_image  = tfmr.coils.combine_coils(complex_tensor, coil_axis=-1)
    return tf.math.abs(complex_image)




def process_prospective_data(raw_data_path, out_path):
    patients = np.unique([pat.split('/')[-1].split('.')[0] for pat in glob.glob(f'{raw_data_path}/*.dat')])
    for patient in patients:
        twixObj =  mapvbvd.mapVBVD(f'{raw_data_path}/{patient}.dat')   
        twixObj = twixObj[1]
        twixObj.image.flagRemoveOS = False
        raw_data = twixObj.image['']
        # raw_data = raw_data[:,:,:,0,:,0,:,0,0,0,0,0,0,0,0,0] 
        raw_data = raw_data[:,:,:,0,0,0,0,0,:,0,0,0,0,0,0,0]
        # if os.path.exists(out_path):
        #         shutil.rmtree(out_path)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        print(raw_data.shape)
        x= process_prospective_kspace(raw_data)
        np.save(f'{out_path}/{patient}.npy', x)
        # for i in range(raw_data.shape[-2]):
        #     print(i)
        #     x= process_prospective_kspace(np.squeeze(raw_data[:,:,:,i,:]))
        #     #display_gifs(np.squeeze(x)[...,0], 'gifs/ani_processed4', f'slice{i}')
        #     np.save(f'{out_path}/{patient}_s{i}.npy', x)



def process_inter4k(raw_data_path, out_path ,config_natural_images=None,config_preproc=None,config_traj=None,reset=1,DEBUG=False):
    """
    Inputs:
    datasets: list of files [train_files,val_files,test_files]
    dataset_dir: path to cached data
    reset: 0,1 or 2 -> 0 resuses cached data if exists / 1 cleans previously cached data after Resizing of Inter4k Data / 2 cleans all previously cached data  
    Returns:
    datasets_withtransforms: tf.Data.Dataset objects with transformation from videos to inputs to FastDVDnet
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if config_preproc is None:
        config_preproc=preproc_multicoil.config_base_preproc()
        # config_preproc['phases'] = 24
        config_preproc['normalize_input'] = True
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],config_preproc['phases'])
        config_natural_images['num_coils'] = [12]
              
    if config_traj is None:
        config_traj=preproc_traj.config_radial_traj()
        # config_traj['phases'] = 24
        config_traj['radial_spokes'] = 10
        config_traj['ordering'] = 'sorted_half'
    

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)
    
    #Define Preprocessing from natural image to kspace to network inputs
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    preproc_function=preproc_multicoil.preprocessing_fn(**config_preproc)
    roll_function=preproc_multicoil.rolling_fn(phases=24,rotation=1,input_format=config_preproc['input_format'])
    traj_function=preproc_traj.create_traj_fn(**config_traj)

    patients = np.unique([pat.split('/')[-1].split('.')[0] for pat in glob.glob(f'{raw_data_path}/*.mp4')])
    for patient in patients[:3]:
      print(patient)
      video = data_util.decode_videos(f'{raw_data_path}/{patient}.mp4')
      video = data_util.resize_video(video)
      kspace = preproc_natural_image(video)
      ds0=tf.data.Dataset.from_tensors(kspace)
      image=traj_function(ds0)
      for element in image:
          zfill, image=preproc_function(element)
          zfill, image=roll_function(zfill, image)
          np.save(f'{out_path}/{patient}_input.npy', zfill)
          np.save(f'{out_path}/{patient}_output.npy', image)

def process_fractals(raw_data_path, out_path ,config_natural_images=None,config_preproc=None,config_traj=None,reset=1,DEBUG=False):
    """
    Inputs:
    datasets: list of files [train_files,val_files,test_files]
    dataset_dir: path to cached data
    reset: 0,1 or 2 -> 0 resuses cached data if exists / 1 cleans previously cached data after Resizing of Inter4k Data / 2 cleans all previously cached data  
    Returns:
    datasets_withtransforms: tf.Data.Dataset objects with transformation from videos to inputs to FastDVDnet
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if config_preproc is None:
        config_preproc=preproc_multicoil.config_base_preproc()
        # config_preproc['phases'] = 24
        config_preproc['normalize_input'] = True
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],config_preproc['phases'])
        config_natural_images['num_coils'] = [10]
        config_natural_images['complex_transform'] = 1
              
    if config_traj is None:
        config_traj=preproc_traj.config_radial_traj()
        # config_traj['phases'] = 24
        config_traj['radial_spokes'] = 13
        config_traj['ordering'] = 'tiny_half'
    

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)
    
    #Define Preprocessing from natural image to kspace to network inputs
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    preproc_function=preproc_multicoil.preprocessing_fn(**config_preproc)
    roll_function=preproc_multicoil.rolling_fn(phases=24,rotation=1,input_format=config_preproc['input_format'], output_format=config_preproc['output_format'])
    traj_function=preproc_traj.create_traj_fn(**config_traj)

    patients = np.unique(['.'.join(pat.split('/')[-1].split('.')[:-1]) for pat in glob.glob(f'{raw_data_path}/*.npy')])
    for patient in patients:
      print(patient)
      # video = data_util.decode_videos(f'{raw_data_path}/{patient}.npy')
      # video = data_util.resize_video(video)
      video = np.load(f'{raw_data_path}/{patient}.npy')
    #   video = make_complex_fractal(video)
      video = np.stack([video.real, video.imag], axis=-1) * 127.5 + 127.5
      video= {'video':tf.cast(tf.clip_by_value(video,0,255),tf.uint8)}
      kspace=preproc_natural_image(video)
      # np.save(f'{out_path}/{patient}.npy', kspace)
      ds0=tf.data.Dataset.from_tensors(kspace)
      image=traj_function(ds0)
      for element in image:
          zfill, image=preproc_function(element)
          zfill, image=roll_function(zfill, image)
          np.save(f'{out_path}/{patient}_input.npy', zfill)
          np.save(f'{out_path}/{patient}_output.npy', image)

def process_fractals_new_grasp(raw_data_path, out_path ,config_natural_images=None,config_preproc=None,config_traj=None,reset=1,DEBUG=False):
    """
    Inputs:
    datasets: list of files [train_files,val_files,test_files]
    dataset_dir: path to cached data
    reset: 0,1 or 2 -> 0 resuses cached data if exists / 1 cleans previously cached data after Resizing of Inter4k Data / 2 cleans all previously cached data  
    Returns:
    datasets_withtransforms: tf.Data.Dataset objects with transformation from videos to inputs to FastDVDnet
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if config_preproc is None:
        config_preproc=preproc_multicoil.config_base_preproc()
        # config_preproc['phases'] = 24
        config_preproc['normalize_input'] = True
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],config_preproc['phases'])
        config_natural_images['num_coils'] = [10]
        config_natural_images['complex_transform'] = 1
              
    if config_traj is None:
        config_traj=preproc_traj.config_radial_traj()
        # config_traj['phases'] = 24
        config_traj['radial_spokes'] = 13
        config_traj['ordering'] = 'golden_half'
    

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)
    
    #Define Preprocessing from natural image to kspace to network inputs
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    preproc_function=preproc_multicoil.preprocessing_fn(**config_preproc)
    roll_function=preproc_multicoil.rolling_fn(phases=24,rotation=1,input_format=config_preproc['input_format'], output_format=config_preproc['output_format'])
    traj_function=preproc_traj.create_traj_fn(**config_traj)

    patients = np.unique(['.'.join(pat.split('/')[-1].split('.')[:-1]) for pat in glob.glob(f'{raw_data_path}/*.npy')])
    for patient in patients:
      print(patient)
      # video = data_util.decode_videos(f'{raw_data_path}/{patient}.npy')
      # video = data_util.resize_video(video)
      video = np.load(f'{raw_data_path}/{patient}.npy')
    #   video = make_complex_fractal(video)
      video = np.stack([video.real, video.imag], axis=-1) * 127.5 + 127.5
      video= {'video':tf.cast(tf.clip_by_value(video,0,255),tf.uint8)}
      kspace=preproc_natural_image(video)
      # np.save(f'{out_path}/{patient}.npy', kspace)
      ds0=tf.data.Dataset.from_tensors(kspace)
      image=traj_function(ds0)
      for element in image:
          zfill, image=preproc_function(element)
          zfill, image=roll_function(zfill, image)
          np.save(f'{out_path}/{patient}_input.npy', zfill)
          np.save(f'{out_path}/{patient}_output.npy', image)

def process_fractals_new_grasp_grouped(raw_data_path, out_path ,config_natural_images=None,config_preproc=None,config_traj=None,seed=0):
    """
    Same as process_fractals_new_grasp, but files are grouped by filename prefix
    (e.g. subj01_slice3.npy -> group 'subj01') and every image in a group gets the
    same random transforms (coils, mask, phase, SNR level, rotation, temporal shift).
    Noise realisation stays independent per image.
    Images within a group must have the same shape.
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if config_preproc is None:
        config_preproc=preproc_multicoil.config_base_preproc()
        config_preproc['normalize_input'] = True
        config_preproc['input_format'] = 'blah'
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],config_preproc['phases'])
        config_natural_images['num_coils'] = [10]
        config_natural_images['complex_transform'] = 1

    if config_traj is None:
        config_traj=preproc_traj.config_radial_traj()
        config_traj['radial_spokes'] = 13
        config_traj['ordering'] = 'golden_half'

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)

    #Define Preprocessing from natural image to kspace to network inputs
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    preproc_function=preproc_multicoil.preprocessing_fn(**config_preproc)
    roll_function=preproc_multicoil.rolling_fn(phases=24,rotation=1,input_format=config_preproc['input_format'], output_format=config_preproc['output_format'])
    traj_function=preproc_traj.create_traj_fn(**config_traj)

    patients = np.unique(['.'.join(pat.split('/')[-1].split('.')[:-1]) for pat in glob.glob(f'{raw_data_path}/*.npy')])
    groups = defaultdict(list)
    for patient in patients:
      groups[patient.rsplit('_', 1)[0]].append(patient)

    for group in sorted(groups):
      group_seed = seed + zlib.crc32(group.encode())
      print(f'Group {group} (seed {group_seed}): {groups[group]}')
      for patient in groups[group]:
        print(patient)
        # Reset generators so every image in the group draws identical transforms
        preprocessing_natural_videos.rg.reset_from_seed(group_seed)
        preproc_multicoil.rg.reset_from_seed(group_seed)
        video = np.load(f'{raw_data_path}/{patient}.npy')
        video = np.stack([video.real, video.imag], axis=-1) * 127.5 + 127.5
        video= {'video':tf.cast(tf.clip_by_value(video,0,255),tf.uint8)}
        kspace=preproc_natural_image(video)
        ds0=tf.data.Dataset.from_tensors(kspace)
        image=traj_function(ds0)
        for element in image:
            zfill, image=preproc_function(element)
            zfill, image=roll_function(zfill, image)
            np.save(f'{out_path}/{patient}_input.npy', zfill)
            np.save(f'{out_path}/{patient}_output.npy', image)
        os.remove(f'{raw_data_path}/{patient}.npy')

def float_to_uint8(array, max, min):
    scaled = (array - min) / (max - min)
    return (scaled * 255.0).astype(int)

def int255_to_float(array, max, min):
    array = array.astype(float)/255
    scaled = array * (max - min) + min
    return scaled

def process_fractals_grasp(raw_data_path, out_path ,config_natural_images=None,config_preproc=None,config_traj=None,reset=1,DEBUG=False):
    """
    Inputs:
    datasets: list of files [train_files,val_files,test_files]
    dataset_dir: path to cached data
    reset: 0,1 or 2 -> 0 resuses cached data if exists / 1 cleans previously cached data after Resizing of Inter4k Data / 2 cleans all previously cached data  
    Returns:
    datasets_withtransforms: tf.Data.Dataset objects with transformation from videos to inputs to FastDVDnet
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if config_preproc is None:
        config_preproc=preproc_multicoil.config_base_preproc()
        config_preproc['phases'] = 10
        config_preproc['normalize_input'] = True
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],320)
        config_natural_images['num_coils'] = [10]
        config_natural_images['complex_transform'] = 1
              
    if config_traj is None:
        config_traj=preproc_traj.config_radial_traj()
        # config_traj['phases'] = 24
        config_traj['phases'] = 10
        config_traj['radial_spokes'] = 20
        config_traj['ordering'] = 'golden_half'
    

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)
    
    #Define Preprocessing from natural image to kspace to network inputs
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    preproc_function=preproc_multicoil.preprocessing_fn(**config_preproc)
    roll_function=preproc_multicoil.rolling_fn(phases=10,roll=0, rotation=0,input_format=config_preproc['input_format'], output_format=config_preproc['output_format'])
    traj_function=preproc_traj.create_traj_fn(**config_traj)

    patients = np.unique(['.'.join(pat.split('/')[-1].split('.')[:-1]) for pat in glob.glob(f'{raw_data_path}/*.npy')])
    for patient in patients[:250]:
      print(patient)
      # video = data_util.decode_videos(f'{raw_data_path}/{patient}.npy')
      # video = data_util.resize_video(video)
      video = np.load(f'{raw_data_path}/{patient}.npy')
      video = make_complex_fractal(video)
      video = tf.transpose(video, [0,3,1,2,4])
      sh = video.shape
      video = tf.reshape(video, [sh[0]*sh[1], sh[2], sh[3], sh[4]])
      video= {'video':tf.cast(tf.clip_by_value(video,0,255),tf.uint8)}
      kspace=preproc_natural_image(video)
    #   print(kspace['kspace'].shape) (1, 32, 10,256,256)
      kspace['kspace'] = tf.reshape(kspace['kspace'], [1, sh[0], sh[1], kspace['kspace'].shape[-3], sh[2], sh[3]])
      full_image = []
      full_zfill = []
      for i in range(sh[1]):  # slices
        slice_kspace = dict(kspace = kspace['kspace'][:,:,i,...])
        print(slice_kspace['kspace'].shape)
        ds0=tf.data.Dataset.from_tensors(slice_kspace)
        image=traj_function(ds0)
        for element in image:
            zfill, image=preproc_function(element)
            zfill, image=roll_function(zfill, image)
            full_image.append(image)
            full_zfill.append(zfill)
      full_image = tf.concat(full_image, axis=-1)
      full_zfill = tf.stack(full_zfill, axis=-2)
    #   full_image_int = float_to_uint8(full_image.numpy(), 1,0)
    #   full_zfill_int = float_to_uint8(full_zfill.numpy(), 1,-1)
    #   re_full_image = int255_to_float(full_image_int,1,0)
    #   re_full_zfill = int255_to_float(full_zfill_int,1,-1)
    #   print(np.mean(full_image - re_full_image))
    #   print(np.mean(full_zfill - re_full_zfill))
      with h5py.File(f'{out_path}/{patient}.h5', 'w') as f:
        f.create_dataset('input', data=full_zfill, compression='gzip')
        f.create_dataset('output', data=full_image, compression='gzip')

def process_fractals_add_phase(raw_data_path, out_path ,config_natural_images=None,config_preproc=None,config_traj=None,reset=1,DEBUG=False):
    """
    Inputs:
    datasets: list of files [train_files,val_files,test_files]
    dataset_dir: path to cached data
    reset: 0,1 or 2 -> 0 resuses cached data if exists / 1 cleans previously cached data after Resizing of Inter4k Data / 2 cleans all previously cached data  
    Returns:
    datasets_withtransforms: tf.Data.Dataset objects with transformation from videos to inputs to FastDVDnet
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if config_preproc is None:
        config_preproc=preproc_multicoil.config_base_preproc()
        # config_preproc['phases'] = 24
        config_preproc['normalize_input'] = True
        config_preproc['output_format'] = None
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],config_preproc['phases'])
        config_natural_images['num_coils'] = [12]
        config_natural_images['complex_transform'] = 1
              
    if config_traj is None:
        config_traj=preproc_traj.config_radial_traj()
        # config_traj['phases'] = 24
        config_traj['radial_spokes'] = 10
        config_traj['ordering'] = 'sorted_half'
    

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)
    
    #Define Preprocessing from natural image to kspace to network inputs
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    preproc_function=preproc_multicoil.preprocessing_fn(**config_preproc)
    roll_function=preproc_multicoil.rolling_fn(phases=24,rotation=1,input_format=config_preproc['input_format'], output_format=config_preproc['output_format'])
    traj_function=preproc_traj.create_traj_fn(**config_traj)

    patients = np.unique(['.'.join(pat.split('/')[-1].split('.')[:-1]) for pat in glob.glob(f'{raw_data_path}/*.npy')])
    for patient in patients:
      print(patient)
      # video = data_util.decode_videos(f'{raw_data_path}/{patient}.npy')
      # video = data_util.resize_video(video)
      video = np.load(f'{raw_data_path}/{patient}.npy')
      video= {'video':tf.cast(tf.clip_by_value(video,0,255),tf.uint8)}
      kspace=preproc_natural_image(video)
      # plot_phase(np.abs(kspace[0]), np.angle(kspace[0]), patient, run)
      # np.save(f'{out_path}/{patient}.npy')
      ds0=tf.data.Dataset.from_tensors(kspace)
      image=traj_function(ds0)
      for element in image:
          zfill, image=preproc_function(element)
          zfill, image=roll_function(zfill, image)
          np.save(f'{out_path}/{patient}_input.npy', zfill)
          np.save(f'{out_path}/{patient}_output.npy', image)

def plot_processed_data(data_path,run):
    patients = np.unique(['_'.join(pat.split('/')[-1].split('_')[:-1]) for pat in glob.glob(f'{data_path}/*.npy')])

    for patient in patients:
      fig, axs = plt.subplots(1, 2, figsize=(6, 3))
      axs[0].set_title(f"True Image")
      axs[1].set_title("Undersampled Image")
      frames = []
      print(patient)
      true_image = np.load(f'{data_path}/{patient}_output.npy')
      noisy_image = np.load(f'{data_path}/{patient}_input.npy')
      noisy_image = convert_multicoil_to_image(noisy_image)

      # Create frames for the animation
      for i in range(true_image.shape[0]):
          # Clear previous titles
          # Set main title for the figure with frame index
          title_text = fig.text(0.5, 0.95, f"Frame {i+1}/{true_image.shape[0]}", ha='center')
          
          # Plot the images for this frame
          p1 = axs[0].imshow(true_image[i,...], cmap='gray', vmin=np.min(true_image), vmax=np.max(true_image))
          p2 = axs[1].imshow(np.squeeze(noisy_image)[i,...], cmap='gray', vmin=np.min(noisy_image), vmax=np.max(noisy_image))
          # Append the current frames to the list
          frames.append([p1, p2, title_text])

      # Create and save animation
      ani = animation.ArtistAnimation(fig, frames)
      Path(f'gifs/processed_data/').mkdir(parents=True, exist_ok=True)
      ani.save(f'gifs/processed_data/{patient}.gif', fps=10)
      # Upload the result to the run object
      run[f'results/processed_data_res/{patient}'].upload(f'gifs/processed_data/{patient}.gif')
      #run[f'results/{cohort}/{patient}'].upload(fig)
      plt.close()  # Close the plot to free memory


# def process_kspace_phase(kspace):
#     """Preprocesses a Wip1200 input.
#     inputs should always have a k-space tensor, although its content differs
#     depending on the context
#         - if `training=True`, `kspace` is a fully-sampled Cartesian *k*-space
#         with shape `(phases, coils, rows, cols)`.
#         - if `training=False`, `kspace` is an undersampled radial *k*-space.
    
#     1) cast as float
#     2) make fs trigcine image
#       -get image via inverse fft
#       -perform augmentation (90rot)
#       -make sure shape is correct
#       -scale image by dividing by max val
#       -find number of phases and save to extra_shape
#     3)compute trajectory and density
#     4) rearrange img matrix to channels,time,height,width and kspace to channels,time,space(CHECK dims)
#     5)expand dims of trajectory and density
#     6)sample non-cartesian kspace using image
#     7)compress coils
#     8)get adjoint to convert back to image space
#     9)get ground truth image from (2)
#     10)combine coils of ground truth image , take abs and expand dims
#     """
#     # This is not used for prospective data
#     kspace = tf.cast(kspace, dtype = tf.complex64)
#     phase_mask = tf.math.reduce_all(tf.abs(kspace) == 0.0, axis=(-2, -1))
#     #kspace = kspace[~phase_mask]
#     kspace = tf.ragged.boolean_mask(kspace, ~phase_mask).to_tensor()
#     kspace = crop_or_circular_pad(kspace, 24)
#     image_shape = tf.constant([256, 256], dtype=tf.int32)
#     image, extra_shape = _make_fs_trigcine_image(kspace, image_shape)
#     trajectory,density = _compute_trajectory_and_density(image_shape, extra_shape)
#     image = tf.einsum('...tchw->...cthw', image) #check dims
#     target_image = image[0] #tfmr.coils.combine_coils(image, coil_axis=-4)
#     # target_image = normalise(target_image)
#     mag = tf.abs(target_image)
#     phase = tf.math.angle(target_image)

#     return (mag, phase)


def plot_phase(mag, phase, patient, run):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].set_title(f"Magnitude")
    axs[1].set_title("Phase")
    frames = []
    # Create frames for the animation
    for i in range(mag.shape[0]):
        # Clear previous titles
        # Set main title for the figure with frame index
        title_text = fig.text(0.5, 0.95, f"Frame {i+1}/{mag.shape[0]}", ha='center')
        
        # Plot the images for this frame
        p1 = axs[0].imshow(mag[i,...], cmap='gray', vmin=np.min(mag), vmax=np.max(mag))
        p2 = axs[1].imshow(np.squeeze(phase)[i,...], cmap='gray', vmin=np.min(phase), vmax=np.max(phase))
        # Append the current frames to the list
        frames.append([p1, p2, title_text])

    # Create and save animation
    ani = animation.ArtistAnimation(fig, frames)
    Path(f'gifs/phase_images/').mkdir(parents=True, exist_ok=True)
    ani.save(f'gifs/phase_images/{patient}.gif', fps=5)
    # Upload the result to the run object
    run[f'results/gifs/{patient}'].upload(f'gifs/phase_images/{patient}.gif')
    #run[f'results/{cohort}/{patient}'].upload(fig)
    plt.close()  # Close the plot to free memory

def make_sine_wave(amplitude=1.0, frequency=1.0, phase=0.0):
    def sine_wave(t):
        return (128*(1 + amplitude * np.sin(2 * np.pi * frequency * t + phase))).astype(int)
    return sine_wave

def gaussian_blur_scipy(image, sigma=1):
    blurred = gaussian_filter(image, sigma=sigma)
    return blurred
  


def sharpen_image(img):
    # 3D Laplacian kernel
    laplacian_kernel = np.array([
        [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 0, 0]]
    ])

    # Apply sharpening
    sharpened_img = convolve(img, laplacian_kernel, mode='reflect')
    return sharpened_img


def central_crop_and_zoom(image, crop_fraction=0.7, order=1):
    """
    Centrally crops a 2D or 3D image and resizes it back to the original shape.
    
    Parameters:
        image (np.ndarray): Input image, shape (H, W) or (H, W, C) or (D, H, W).
        crop_fraction (float): Fraction to crop (e.g., 0.8 crops to 80% in each dimension).
        order (int): Interpolation order (0=nearest, 1=bilinear/trilinear, 3=cubic).
    
    Returns:
        np.ndarray: Cropped and zoomed image with same shape as input.
    """
    original_shape = np.array(image.shape)
    crop_shape = (original_shape * crop_fraction).astype(int)
    
    # Calculate crop slices
    start = ((original_shape - crop_shape) // 2).astype(int)
    end = start + crop_shape
    
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    cropped = image[slices]
    
    # Resize back to original shape
    zoomed = sc_resize(cropped, original_shape, order=order, mode='reflect', anti_aliasing=True)
    return zoomed

def make_complex_fractal(ex):
    wave1 = make_sine_wave(amplitude=1, 
                        frequency=0.25, 
                        phase=np.random.uniform(0,1))
    wave2 = make_sine_wave(amplitude=1, 
                        frequency=0.25, 
                        phase=np.random.uniform(0,1))
    ex = norm(ex)
    ex = gaussian_blur_scipy(ex.astype(float), sigma=0.8)
    real = wave1(ex)
    imag = wave2(ex)
    img = np.stack([real, imag], axis=-1)
    return img

def make_complex_fractal_spiral(ex):
    wave1 = make_sine_wave(amplitude=1, 
                        frequency=np.random.uniform(0.2,0.5), 
                        phase=np.random.uniform(0,1))
    temp = gaussian_blur_scipy(ex, sigma=0.8)
    temp = wave1(norm(temp))
    real = temp
    imag = temp
    img = np.stack([real, imag], axis=-1)
    return img

def norm(x):
    return(x.max()-x)/(x.max()-x.min())



def median_blur_3d_slicewise(volume: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median blur to a 3D image slice-wise (along axis 0).

    Parameters:
        volume (np.ndarray): 3D input array of shape (D, H, W)
        kernel_size (int): Size of the median filter kernel (must be odd)

    Returns:
        np.ndarray: Blurred 3D image of same shape
    """
    assert volume.ndim == 3, "Input must be a 3D array"
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    blurred = np.empty_like(volume)

    for i in range(volume.shape[0]):
        blurred[i] = median_filter(volume[i], size=kernel_size)

    return blurred

def unsharp_mask_3d_slicewise(volume: np.ndarray, alpha: float = 1.0, blur_sigma: float = 1.0) -> np.ndarray:
    """
    Apply slice-wise unsharp masking to a 3D volume.

    Parameters:
        volume (np.ndarray): 3D array of shape (D, H, W)
        alpha (float): Sharpening strength
        blur_sigma (float): Sigma for Gaussian blur

    Returns:
        np.ndarray: Sharpened 3D volume
    """
    assert volume.ndim == 3, "Input must be a 3D array"

    sharpened = np.empty_like(volume)

    for i in range(volume.shape[0]):
        blurred_slice = gaussian_filter(volume[i], sigma=blur_sigma)
        mask = volume[i] - blurred_slice
        sharpened[i] = volume[i] + alpha * mask

    return sharpened


def process_fractals_spiral(raw_data_path, out_path ,config_natural_images=None,config_preproc=None,config_traj=None,reset=1,DEBUG=False):
    """
    Inputs:
    datasets: list of files [train_files,val_files,test_files]
    dataset_dir: path to cached data
    reset: 0,1 or 2 -> 0 resuses cached data if exists / 1 cleans previously cached data after Resizing of Inter4k Data / 2 cleans all previously cached data  
    Returns:
    datasets_withtransforms: tf.Data.Dataset objects with transformation from videos to inputs to FastDVDnet
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if config_preproc is None:
        config_preproc=preproc_fastdvdnet.config_base_preproc()
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],config_preproc['phases'])
              
    if config_traj is None:
        config_traj=preproc_traj.config_optimized_traj()
    

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)
    
    #Define Preprocessing from natural image to kspace to network inputs
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    preproc_function=preproc_fastdvdnet.preprocessing_fn(**config_preproc)
    roll_function=preproc_fastdvdnet.rolling_fn()
    traj_function=preproc_traj.create_traj_fn(**config_traj)
    patients = np.unique(['.'.join(pat.split('/')[-1].split('.')[:-1]) for pat in glob.glob(f'{raw_data_path}/*.npy')])
    for i, patient in enumerate(patients[:1]):
      print(patient)
      # video = data_util.decode_videos(f'{raw_data_path}/{patient}.npy')
      # video = data_util.resize_video(video)
      video = np.load(f'{raw_data_path}/{patient}.npy')
      video = make_complex_fractal_spiral(video)
      print(video.max(), video.min())
      video= {'video':tf.cast(tf.clip_by_value(video,0,255),tf.uint8)}
      kspace=preproc_natural_image(video)
      # np.save(f'{out_path}/{patient}.npy', kspace)
      ds0=tf.data.Dataset.from_tensors(kspace)
      image=traj_function(ds0)
      for element in image:
          zfill, image=preproc_function(element)
          zfill, image=roll_function(zfill, image)
          np.save(f'{out_path}/{patient}_input.npy', zfill)
          np.save(f'{out_path}/{patient}_output.npy', (1-equalize_adapthist(norm(image.numpy()))))



def process_prospective_spiral(raw_data_path, out_path):
    patients = np.unique([pat.split('/')[-1].split('.')[0] for pat in glob.glob(f'{raw_data_path}/*')])
    for patient in patients:
      kspace,imparams=recon_utils.load_data(f'{raw_data_path}/{patient}.dat',model_type='FastDVDNet',method='focus')
      temptraj=dict()
      startbatch=3
      nbatches=8
      timepoints=[startbatch*15-4,(startbatch+nbatches-1)*15+24]#Grid all data then run FastDVDnet
      # if speeches[nfile]==1:
      #     timepoints=[1*15-4,kspace.shape[1]]#Grid all data then run FastDVDnet
      uskspace=tf.transpose(kspace[0,timepoints[0]:timepoints[1],...],perm=[0,3,1,2])
      trajectory=recon_utils.get_trajectory(tf.transpose(kspace[0,...],perm=[0,3,1,2]),'FastDVDNet',imparams)
      temptraj['traj']=trajectory['traj'][timepoints[0]:timepoints[1],...]
      temptraj['dcw']=trajectory['dcw'][timepoints[0]:timepoints[1],...]
      #uskspace=recon_utils.offcenterspiralcorrection(uskspace[...,:temptraj['traj'].shape[-2]],temptraj['traj'],-imparams['Imaging_Position'],imparams['CalculatedPixSize'])
      image_shape=[int(imparams['lBaseResolution']),int(imparams['lBaseResolution'])]

      inputs=recon_utils.prep_network_data_radial(uskspace,temptraj,image_shape=image_shape,input_format=None,return_kspace=False)

      inputs_reformat=tfmr.coils.combine_coils(tf.complex(inputs[...,:inputs.shape[-1]//2],inputs[...,inputs.shape[-1]//2:]),coil_axis=-1)
      inputs_reformat=tf.transpose(inputs_reformat[...],perm=[0,2,3,1])
      print(inputs_reformat.shape)
      Path(out_path).mkdir(parents=True, exist_ok=True)
      np.save(f'{out_path}/{patient}', inputs_reformat)



def process_prospective_cartesian(raw_data_path, out_path):
    shutil.rmtree(out_path)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    startbatch=3
    nbatches=8
    out_labels = ['u_t', 'f', 'coil_sens', 'mask']
    patients = np.unique([pat.split('/')[-1].split('.')[0] for pat in glob.glob(f'{raw_data_path}/*')])
    for patient in patients:
        print(patient)
        kspace,imparams=recon_utils.load_data(f'{raw_data_path}/{patient}.dat',model_type='VarNet')
        for batch in range(nbatches):
            timepoints=[(startbatch+batch)*15,(startbatch+batch)*15+24]
            uskspace0=np.transpose(kspace[0,timepoints[0]:timepoints[1],...],[0,3,1,2])
            inputs=recon_utils.prep_network_data_cartesian(uskspace0)
            with h5py.File(f'{out_path}/{patient}.h5', 'a') as f:
                for key, tensor in zip(out_labels, inputs):
                    f.create_dataset(f'{key}_{batch}', data=tensor.numpy())
        


def generate_dataset_VarNet(raw_data_path, out_path,config_natural_images=None,config_preproc=None,config_traj=None,reset=1,DEBUG=False):
    """
    Inputs:
    datasets: list of files [train_files,val_files,test_files]
    dataset_dir: path to cached data
    reset: 0,1 or 2 -> 0 resuses cached data if exists / 1 cleans previously cached data after Resizing of Inter4k Data / 2 cleans all previously cached data  
    Returns:
    datasets_withtransforms: tf.Data.Dataset objects with transformation from videos to inputs to FastDVDnet
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if config_preproc is None:
        config_preproc=preproc_varnet.config_base_preproc()
    if config_natural_images is None:
        config_natural_images=preprocessing_natural_videos.config_default(config_preproc['base_resolution'],config_preproc['phases'])
              
    if config_traj is None:
        config_traj={'acc':14,
        'phases':15,
        'center_lines':8,
        'half_fourier':0.6,
        'mask_type':'noreselect'}
    

    print('Config natural image to kspace:',config_natural_images,
              '\nConfig preprocessing:',config_preproc,
              '\nConfig traj:',config_traj)
    
    #Define Preprocessing from natural image to kspace to network inputs
    seed_value=7
    mask=generate_mask(seed_value,(config_traj['phases'],1,config_preproc['base_resolution'],config_preproc['base_resolution'],1),[config_traj['acc']],[config_traj['center_lines']],half_fourier=config_traj['half_fourier'],mask_type=config_traj['mask_type'])     
    preproc_natural_image=preprocessing_natural_videos.preprocessing_fn(**config_natural_images)
    patients = np.unique(['.'.join(pat.split('/')[-1].split('.')[:-1]) for pat in glob.glob(f'{raw_data_path}/*.npy')])
    preproc_function=preproc_varnet.preprocessing_fn(**config_preproc)
    out_labels = ['u_t', 'f', 'coil_sens', 'mask', 'target']
    for patient in patients:
        video = np.load(f'{raw_data_path}/{patient}.npy')
        video = make_complex_fractal(video)
        video= {'video':tf.cast(tf.clip_by_value(video,0,255),tf.uint8)}
        kspace=preproc_natural_image(video)
        preproc_dict = {'image':kspace, 'trajectory': {'mask':mask}}
        out_list = preproc_function(preproc_dict)
        with h5py.File(f'{out_path}/{patient}.h5', 'w') as f:
            for key, tensor in zip(out_labels, out_list):
                f.create_dataset(key, data=tensor.numpy())

        
        
        


if __name__ == "__main__":
    # shutil.rmtree('resources/processed_data_original', ignore_errors=True)
    # shutil.rmtree('resources/processed_data_time_superres', ignore_errors=True)
    # process_data('resources/cine_phase_test' , 'resources/cmri_coil_images')
    # process_superres_data('resources/rfh_cine_mri_subset' , 'resources/processed_data_time_superres')
    # shutil.rmtree('resources/processed_data_inter4k', ignore_errors=True)
    # model_name = 'UN2-212'
    # run = neptune.init_run(
    #     project="my-learning/unet3plus-tests",
    #     api_token="<REDACTED>",
    #     with_id = model_name
    #     )
    # shutil.rmtree('resources/processed_3Djulia_flow')
    # process_fractals_add_phase('resources/3Djulias', 'resources/processed_3Djulia_flow')
    # plot_processed_data('resources/processed_3Djulia_flow', run)
    # process_inter4k('resources/inter4k_raw','resources/inter4k_phase')
    # process_fractals('resources/3Djulias', 'resources/julia_phase')
    # plot_phase_data('resources', 'resources/phase_im', run)
    # shutil.rmtree('resources/julia_radial', ignore_errors=True)
    # process_fractals_new_grasp('resources/new_julia_grasp', 'resources/julia_grasp_processed')
    process_fractals_new_grasp_grouped('resources/new_julia_grasp', 'resources/julia_grasp_processed')
    # shutil.rmtree('resources/julia_varnet', ignore_errors=True)
    # generate_dataset_VarNet('resources/3Djulias', 'resources/julia_varnet')
    # process_prospective_spiral('resources/spiral_prospective', 'resources/processed_spiral_prospective')
    # process_prospective_data('resources/13_spokes_prospective', 'resources/processed_13_spokes_prospective')
    # shutil.rmtree('resources/processed_fractals_grasp', ignore_errors=True)
    # process_fractals_grasp('resources/4Djulias', 'resources/processed_fractals_grasp')
    # folder_path = ''

    # # Find all .npy files in the folder
    # npy_files = glob.glob(os.path.join(folder_path, '*.h5'))

    # # Delete each file
    # for file in npy_files:
    #     try:
    #         os.remove(file)
    #         print(f"Deleted: {file}")
    #     except OSError as e:
    #         print(f"Error deleting {file}: {e}")