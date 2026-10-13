
"""Noise and coordinate-domain transforms adapted from the notebook."""

import numpy as np


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _gradient_noise_3d(x, y, z, seed=0):
    rng = np.random.RandomState(seed)
    perm = np.concatenate([rng.permutation(256), rng.permutation(256)]).astype(np.int32)
    grad3 = np.array([
        [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
        [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
        [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],
    ], dtype=np.float64)

    xi, yi, zi = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), np.floor(z).astype(np.int32)
    xf, yf, zf = x-xi, y-yi, z-zi
    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    def hash3(ix, iy, iz):
        return perm[(perm[(perm[ix & 255] + iy) & 255] + iz) & 255] % 12

    def dot(ix, iy, iz, dx, dy, dz):
        g = grad3[hash3(ix, iy, iz)]
        return g[...,0]*dx + g[...,1]*dy + g[...,2]*dz

    n000=dot(xi,yi,zi,xf,yf,zf)
    n100=dot(xi+1,yi,zi,xf-1,yf,zf)
    n010=dot(xi,yi+1,zi,xf,yf-1,zf)
    n110=dot(xi+1,yi+1,zi,xf-1,yf-1,zf)
    n001=dot(xi,yi,zi+1,xf,yf,zf-1)
    n101=dot(xi+1,yi,zi+1,xf-1,yf,zf-1)
    n011=dot(xi,yi+1,zi+1,xf,yf-1,zf-1)
    n111=dot(xi+1,yi+1,zi+1,xf-1,yf-1,zf-1)
    lerp=lambda a,b,t: a+t*(b-a)
    nx00=lerp(n000,n100,u); nx10=lerp(n010,n110,u)
    nx01=lerp(n001,n101,u); nx11=lerp(n011,n111,u)
    return lerp(lerp(nx00,nx10,v), lerp(nx01,nx11,v), w)


def fractal_noise_3d(x, y, z, octaves=4, persistence=0.5, lacunarity=2.0, seed=0):
    """Fractional Brownian-motion sum of 3-D gradient-noise octaves."""
    total=np.zeros_like(x,dtype=np.float64)
    amp=1.0
    freq=1.0
    max_amp=0.0
    for octave in range(octaves):
        total += amp * _gradient_noise_3d(
            x*freq,y*freq,z*freq,seed=seed+101*octave
        )
        max_amp += amp
        amp *= persistence
        freq *= lacunarity
    return total / max_amp


def domain_warp(coords, strength=0.3, scale=1.0, octaves=4,
                persistence=0.5, lacunarity=2.0, seed=0):
    """Warp quaternion coordinates with four correlated smooth noise fields.

    ``coords`` has shape ``(..., 4)`` and uses ``(w, x, y, z)`` ordering.
    """
    coords=np.asarray(coords,dtype=np.float64)
    w,x,y,z=np.moveaxis(coords,-1,0)
    sx,sy,sz=x*scale,y*scale,z*scale
    sw=w*scale
    dw=fractal_noise_3d(sx,sy,sz+sw,octaves,persistence,lacunarity,seed)
    dx=fractal_noise_3d(sy,sz,sx+sw+1000,octaves,persistence,lacunarity,seed+1)
    dy=fractal_noise_3d(sz,sx,sy+sw+2000,octaves,persistence,lacunarity,seed+2)
    dz=fractal_noise_3d(sx,sz,sy+sw+3000,octaves,persistence,lacunarity,seed+3)
    out=coords.copy()
    out[...,0]=w+strength*dw
    out[...,1]=x+strength*dx
    out[...,2]=y+strength*dy
    out[...,3]=z+strength*dz
    return out
