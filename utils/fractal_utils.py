import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter

def make_sine_wave(amplitude=1.0, frequency=1.0, phase=0.0):
    """
    Returns a function that computes a sine wave with specified amplitude, frequency, and phase.
    
    Parameters:
        amplitude (float): Amplitude of the wave
        frequency (float): Frequency in Hz
        phase (float): Phase in radians

    Returns:
        sine_wave (function): Function that takes a time array and returns the wave values
    """
    def sine_wave(t):
        return (128*(1 + amplitude * np.sin(2 * np.pi * frequency * t + phase))).astype(int)
    
    return sine_wave


@njit
def quaternion_multiply_scalar(q1, q2):
    """
    Returns the product of two quaternions.
    
    Parameters:
        q1 (np.array): quaternion 1
        q2 (np.array): quaternion 2

    Returns:
        np.array: q1*q2
    """
     
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        x1*x2 - y1*y2 - z1*z2 - w1*w2,
        x1*y2 + y1*x2 + z1*w2 - w1*z2,
        x1*z2 - y1*w2 + z1*x2 + w1*y2,
        x1*w2 + y1*z2 - z1*y2 + w1*x2
    ])



@njit
def quaternion_abs_scalar(q):
    """
    Returns the scalar magnitude of a quaternion
    
    Parameters:
        q (np.array): quaternion number to check

    Returns:
        float: ||q||
    """
    return np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)

@njit(parallel=True)
def julia_escape(grid, c, max_iter=50, escape_radius=4.0):
    """
    Returns the escape time matrix for a Julia set the across the specified grid and with the specified c value
    
    Parameters:
        grid (np.array): spatial grid across which to generate the Julia set
        c (np.array): quaternion Julia structure constant
        max_iter (int): maximum escape time to check for
        escape_radius (float): escape threshold radius

    Returns:
        np.array: Julia escape times across the input grid
    """
    shape = grid.shape[:-1]
    escape_times = np.full(shape, max_iter, dtype=np.int32)
    c = np.broadcast_to(c, grid.shape)

    Nx, Ny, Nz, Nw = shape

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nw):
                    q = grid[i, j, k, l]
                    c_val = c[i, j, k, l]
                    for it in range(max_iter):
                        q = quaternion_multiply_scalar(q,q) + c_val
                        if quaternion_abs_scalar(q) > escape_radius:
                            escape_times[i, j, k, l] = it
                            break

    return escape_times

@njit(parallel=True)
def mandelbrot_escape(c_grid, max_iter=50, escape_radius=4.0):
    """
    Returns the escape time matrix for a Mandelbrot set the across the specified grid and with the specified c value
    
    Parameters:
        c_grid (np.array): spatial grid across which to generate the Mandelbrot set
        max_iter (int): maximum escape time to check for
        escape_radius (float): escape threshold radius

    Returns:
        np.array: Mandelbrot escape times across the input grid
    """
    shape = c_grid.shape[:-1]
    escape_times = np.full(shape, max_iter, dtype=np.int32)
    Nx, Ny, Nz, Nw = shape

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nw):
                    c = c_grid[i, j, k, l]
                    q = np.zeros(4, dtype=np.float64)
                    for it in range(max_iter):
                        q = quaternion_multiply_scalar(q, q)
                        q = q + c  # add elementwise
                        if quaternion_abs_scalar(q) > escape_radius:
                            escape_times[i, j, k, l] = it
                            break
    return escape_times

def build_grid(limits, sizes):
    """
    Returns a grid with specified spatial extent and number of points
    
    Parameters:
        limits (list): list of positive extents of each dimension
        sizes (list): list of the number of points in each dimension

    Returns:
        np.array: A grid of shape tuple(sizes) and max and min values of limits and -limits in each dimension
    """
    coords = []
    for lim, size in zip(limits, sizes):
        coords.append(np.linspace(-lim,lim,size))
    grd = list(np.meshgrid(*coords, indexing='ij'))
    return np.stack(grd, axis=-1)

def test_complexity(fractal, threshold=1.7):
    """
    Tests the entropy of a fractal array 
    
    Parameters:
        fractal (np.array): Grid populated with fractal escape times
        threshold (float): Entropy threshold to check for (default: 1.7)

    Returns:
        bool: True if entropy is above the threshold else False
    """
    hist, _ = np.histogram(fractal, bins=len(np.unique(fractal)))
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log(prob + 1e-9))
    if entropy < threshold:
        return False 
    return True


def norm(x):
    """
    Fast safe normalization.
    """
    xmin = x.min()
    xmax = x.max()

    if xmax == xmin:
        return np.zeros_like(x, dtype=np.float32)

    return (x - xmin) / (xmax - xmin)



def blur_and_sharpen(image, sigma=1.0, alpha=50.0):
    """
    Optimised unsharp mask.
    """
    base = gaussian_filter(image, sigma)
    detail = base - gaussian_filter(base, 0.1)
    return base + alpha * detail



def make_complex_fractal(ex):
    """
    Convert escape-time data into fractal texture.
    """

    ex = norm(ex)

    freq1 = np.random.uniform(0.25, 1.0)
    freq2 = 0.25

    if np.random.rand() > 0.5:
        freq1, freq2 = freq2, freq1

    wave1 = make_sine_wave(1, freq1, np.random.uniform(0, 2))
    wave2 = make_sine_wave(1, freq2, np.random.uniform(0, 2))

    real = blur_and_sharpen(wave1(ex), np.random.uniform(0.2, 0.4))
    imag = blur_and_sharpen(wave2(ex), np.random.uniform(0.2, 0.4))

    return np.stack((real, imag), axis=-1)




if __name__=='__main__':
    C_grid = build_grid([1.5]*4, [100]*4)
    mandelbrot_vals = mandelbrot_escape(C_grid, max_iter=50)
    ids = np.argwhere((mandelbrot_vals<30) & (mandelbrot_vals>10))
    C_candidates = C_grid[tuple(ids.T)]

    grid1 = build_grid(limits=[1,1,0,0.2], sizes=[256,256,1,32])
    i=0
    while i < 1:
        # print(i)
        idx = np.random.randint(0,C_candidates.shape[0])
        c = C_candidates[idx]
        escape_vals = np.squeeze(julia_escape(grid1, c, max_iter=50))
        if test_complexity(escape_vals):
            complex_fractal = make_complex_fractal(np.transpose(escape_vals,[2, 0, 1]))
            i+=1
            np.save('test4', complex_fractal)
            print('done')