"""TF implementation of trajectories and the sampling operator.

``gen_spiral_traj`` and the spiral search loop are verbatim from the legacy
``preprocessing_trajectory_gen.py``.
"""


import tensorflow as tf
import tensorflow_mri as tfmri
import tensorflow_nufft as tfft

def gen_spiral_traj(
    base_resolution=256,
	field_of_view=400,phases=1,
	ordering='golden',tiny_number=7,
	max_tempres=50,min_max_arm_time=[3,8],
	vd_spiral_arms=26,vd_inner_cutoff=0.2,
	pre_vd_outer_cutoff=0.5,vd_outer_density=0.25,
	vd_type='linear',
    max_grad_ampl=24.0,
    min_rise_time=8.125,
    dwell_time=1.4,
    gradient_delay=0.56,readoutOS=2,deadtime=1.9,larmor_const=42.577478518,flow=0):
        """Returns a spiral trajectory.
        Computes 1 arm, if conditions min_max_arm_time respected:
            computes the full trajectory -> (nphases, narms, nsamples, 2)
            and density compensation weights -> (nphases, narms, nsamples)
            and time_for_an_arm -> float
        """
        views=1
        if vd_inner_cutoff>0.9:
            vd_outer_cutoff=1
            vd_outer_density=1
        else:
            vd_outer_cutoff=float(vd_inner_cutoff+0.1+pre_vd_outer_cutoff*(1-vd_inner_cutoff-0.1))
        #Generate 1 spiral arm
        traj_var=tfmri.spiral_trajectory(
            base_resolution, vd_spiral_arms, field_of_view, 
            max_grad_ampl, min_rise_time, dwell_time,
            views=1,phases=1,ordering=ordering,
            tiny_number=tiny_number,
            vd_inner_cutoff=vd_inner_cutoff,
            vd_outer_cutoff=vd_outer_cutoff,
            vd_outer_density=vd_outer_density,
            gradient_delay=gradient_delay,vd_type=vd_type,readout_os=readoutOS,larmor_const=larmor_const)
        
        # If within the specified time for an arm allowed generates the whole trajectory otherwise returns incomplete traj

        time_for_an_arm=traj_var.shape[2]*dwell_time*10**-3
        dcwvar=tf.ones((1,1,1))

        if (time_for_an_arm<min_max_arm_time[1] and time_for_an_arm>min_max_arm_time[0]) or vd_spiral_arms==1:
            
            TR=time_for_an_arm+deadtime
            if flow:
                #For Flow more time for velocity encoding ->+3.6
                TR=time_for_an_arm+3.6
            
            #determine number of spiral arms depending on max temporal resolution and calculated TR.
            views=int(max_tempres/TR)
            #Generate trajectory
            traj_var=tfmri.spiral_trajectory(
                base_resolution, vd_spiral_arms, field_of_view, 
                max_grad_ampl, min_rise_time, dwell_time,
                views=views,phases=phases,ordering=ordering,
                tiny_number=tiny_number,
                vd_inner_cutoff=vd_inner_cutoff,
                vd_outer_cutoff=vd_outer_cutoff,
                vd_outer_density=vd_outer_density,
                gradient_delay=gradient_delay,vd_type=vd_type,readout_os=readoutOS,larmor_const=larmor_const)
            
            #Density Estimation
            traj_var_full=tfmri.spiral_trajectory(
                base_resolution, vd_spiral_arms, field_of_view, 
                max_grad_ampl, min_rise_time, dwell_time,
                ordering=ordering,tiny_number=tiny_number,
                views=vd_spiral_arms,
                vd_inner_cutoff=vd_inner_cutoff,
                vd_outer_cutoff=vd_outer_cutoff,
                vd_outer_density=vd_outer_density,
                gradient_delay=gradient_delay,vd_type=vd_type,readout_os=readoutOS,larmor_const=larmor_const)
            #truncate center point (gradient delay)
            traj_var=traj_var[:,:,1:,:]
            traj_var_full=traj_var_full[:,1:,:]
            #Reshaping trajectory:
            var_shape=traj_var.shape
            var_shape_full=traj_var_full.shape
            traj_var_full=tf.reshape(traj_var_full, (1,var_shape_full[0]* var_shape_full[1], var_shape_full[2]))
            dens=tfmri.estimate_density(traj_var_full[0:1,...], [base_resolution,base_resolution],method='pipe',max_iter=30)
            dcwvarfull=tf.math.divide_no_nan(1.0,dens)
            traj_var_full=tf.reshape(traj_var_full, var_shape_full)
            dcwvarfull=tf.reshape(dcwvarfull,var_shape_full[0:-1])
            dcwvarfull=tf.tile(tf.expand_dims(dcwvarfull,0),(var_shape[0],)+(1,1,))
            
            if views>vd_spiral_arms:
                dcwvarfull=tf.tile(dcwvarfull,(1,views,1,))

            dcwvar=dcwvarfull[:,0:views,...]
        
        return traj_var, dcwvar, time_for_an_arm

def radial(cfg):
    """Radial trajectory ``(phases, spokes*readout, 2)`` and dcw ``(phases, spokes*readout)``."""
    traj = tfmri.sampling.radial_trajectory(base_resolution=cfg.base_resolution, views=cfg.radial_spokes,
                                            phases=cfg.phases, ordering=cfg.ordering)
    traj = tfmri.sampling.flatten_trajectory(traj)
    dens = tfmri.sampling.estimate_radial_density(traj)
    return traj, tf.math.divide_no_nan(1.0, dens)


def spiral(cfg):
    """Variable-density spiral, searching the arm count to satisfy ``min_max_arm_time`` (legacy)."""
    p = cfg.spiral
    min_max_arm_time = p["min_max_arm_time"]
    max_guesses, counter, time_for_an_arm = 50, 0, 0
    vd_spiral_arms0 = p["vd_spiral_arms"]
    traj = dcw = None
    while not (min_max_arm_time[0] < time_for_an_arm < min_max_arm_time[1]) and counter < max_guesses and vd_spiral_arms0 > 0:
        traj, dcw, time_for_an_arm = gen_spiral_traj(
            base_resolution=cfg.base_resolution, field_of_view=cfg.field_of_view, phases=cfg.phases,
            ordering=cfg.ordering, **{**p, "vd_spiral_arms": vd_spiral_arms0})
        counter += 1
        if time_for_an_arm < min_max_arm_time[0]:
            vd_spiral_arms0 -= 1
        elif time_for_an_arm > min_max_arm_time[1]:
            vd_spiral_arms0 += 1
    return traj, dcw


def make_trajectory(cfg):
    """``(traj, dcw)`` for a :class:`synthmr.sampling.TrajectoryConfig` (in-plane part)."""
    if cfg.kind == "radial":
        traj, dcw = radial(cfg)
    elif cfg.kind == "spiral":
        traj, dcw = spiral(cfg)
    else:
        raise ValueError(f"unknown trajectory kind {cfg.kind!r}")
    if cfg.reverse:
        traj = tf.reverse(traj, [-1])
    return traj, dcw


def tile_to_phases(traj, dcw, phases):
    """Repeat a trajectory along time to cover ``phases`` frames (legacy ``reps`` tiling)."""
    reps = tf.cast(tf.math.ceil(phases / tf.shape(traj)[0]), tf.int32)
    return tf.tile(traj, [reps, 1, 1])[:phases, ...], tf.tile(dcw, [reps, 1])[:phases, ...]


class SamplingOperator:
    """Non-Cartesian sampling of multicoil image series.

    Through-plane encoders:
      ``none``          2D: image ``(T, C, H, W)`` <-> k-space ``(T, C, R)``.
      ``cartesian_kz``  stack-of-*: image ``(T, C, Z, H, W)``; centred ortho FFT along Z, then the
                        in-plane trajectory per kz partition -> k-space ``(T, C, Z, R)``.
    Forward and adjoint include the legacy ``1/sqrt(H*W)`` scaling; density compensation is
    applied separately (``apply_dcw``) as in the legacy pipeline.
    """

    def __init__(self, traj, dcw, image_shape, through_plane="none", kz_traj=None):
        self.traj, self.dcw = traj, dcw
        self.image_shape = list(image_shape)
        self.through_plane = through_plane
        # Optional per-partition trajectories (T, Z, R, 2) e.g. rotated stars; default: same star for all kz.
        self.kz_traj = kz_traj
        self._npixel = tf.cast(self.image_shape[0] * self.image_shape[1], tf.complex64)

    def _points(self, ndim_extra):
        if self.kz_traj is not None:
            return tf.expand_dims(self.kz_traj, axis=1)                     # (T, 1, Z, R, 2)
        t = tf.expand_dims(self.traj, axis=-3)                              # (T, 1, R, 2)
        return tf.expand_dims(t, axis=-3) if ndim_extra else t             # (T, 1, 1, R, 2)

    def forward(self, image):
        image = tf.cast(image, tf.complex64)
        kz = self.through_plane == "cartesian_kz"
        if kz:
            image = tfmri.signal.fft(image, axes=[-3], norm='ortho', shift=True)
        return tfft.nufft(image, self._points(kz), transform_type='type_2',
                          fft_direction='forward') / tf.sqrt(self._npixel)

    def adjoint(self, kspace):
        kz = self.through_plane == "cartesian_kz"
        image = tfft.nufft(kspace, self._points(kz), grid_shape=self.image_shape, transform_type='type_1',
                           fft_direction='backward') / tf.sqrt(self._npixel)
        if kz:
            image = tfmri.signal.ifft(image, axes=[-3], norm='ortho', shift=True)
        return image

    def apply_dcw(self, kspace):
        dcw = tf.cast(self.dcw, tf.complex64)
        if self.through_plane == "cartesian_kz":
            return kspace * dcw[:, None, None, :]
        return kspace * tf.expand_dims(dcw, axis=-2)
