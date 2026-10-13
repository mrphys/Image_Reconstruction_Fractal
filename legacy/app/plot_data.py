import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import tensorflow_mri as tfmr



def convert_to_complex(stacked_tensor):
    real_part, imag_part = tf.split(stacked_tensor, 2, axis=-1)
    complex_tensor = tf.complex(real_part, imag_part)
    return complex_tensor

def convert_multicoil_to_image(multicoil):
    complex_tensor = convert_to_complex(multicoil)
    complex_image  = tfmr.coils.combine_coils(complex_tensor, coil_axis=-1)
    return tf.math.abs(complex_image)

X = []
for i in range(1,9):
    # Load the data
    x = np.load(f'resources/julia_grasp_processed/julia_01_slice_{i}_input.npy')
    X.append(x)



X = tf.stack(X, axis=1)

noisy_image = convert_multicoil_to_image(X).numpy()
print(noisy_image.shape)

noisy_lims = (np.min(noisy_image), np.max(noisy_image))


# One gif per slice, animated over time
for s in range(8):
    fig, axs = plt.subplots(1, 1, figsize=(3, 3))
    axs.set_title("Undersampled Image")
    frames = []

    # Create frames for the animation
    for t in range(20):
        p3 = axs.imshow(noisy_image[t, s], cmap='gray', vmin=noisy_lims[0], vmax=noisy_lims[1])
        # Append the current frames to the list
        frames.append([p3])

    ani = animation.ArtistAnimation(fig, frames)
    gif_path = f'test_slice_{s+1}.gif'
    ani.save(gif_path, fps=20)
