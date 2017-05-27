import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)


class Series:
    """A class which contains one or more dataset objects"""
    datasets = []

    def __init__(self, file_list, pixel_list, particle_size_list, frame_time_list):
        for i in range(len(file_list)):
            self.datasets.append(DDMDataset(file_list[i], pixel_list[i], particle_size_list[i], frame_time_list[i]))
        self.length = len(self.datasets)


class DDMDataset:
    """A class which contains a single DDM dataset. The class has properties which describe the dataset"""

    def __init__(self, filename, pixel_size, particle_size, frame_time):
        self.data = np.loadtxt(filename)
        self.pixel_size = np.array(pixel_size)
        self.particle_size = np.array(particle_size)
        self.pixels_per_particle = self.particle_size / self.pixel_size
        self.frame_time = np.array(frame_time)


def plot_wavevector_range(ddm_dataset, wavevectors):
    """Accepts a dataset object as input. Plots magnitude of Fourier difference at a range of different wavevectors for a single sample."""

    minwavevector = wavevectors[0]
    maxwavevector = wavevectors[1]
    wavevectorstep = 1

    for i in range(minwavevector, maxwavevector, wavevectorstep):
        wavevector = '{:1.2f}'.format(i / ddm_dataset.pixels_per_particle)
        plt.semilogx([j * ddm_dataset.frame_time for j in range(ddm_dataset.data.shape[0])], ddm_dataset.data[:, i], label=wavevector)

    plt.ylabel(r'', fontsize=16)
    plt.xlabel(r'Time (s)', fontsize=16)
    # plt.ylim(0, 5E8)
    # plt.xlim(1, 200)
    plt.legend(bbox_to_anchor = (1.5,1.5))
    # plt.savefig("wavevectors.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_wavevector_samples(data_series, wavevector):
    """Plots magnitude of Fourier difference at a single wavevector for multiple samples."""

    for i in range(0, data_series.length):
        current_dataset = data_series.datasets[i]
        plt.semilogx([i * current_dataset.frame_time for i in range(current_dataset.data.shape[0])],
                     current_dataset.data[:, wavevector], label=str(current_dataset.pixel_size) + " nm px, q="
                    + '{:1.2f}'.format(wavevector / current_dataset.pixels_per_particle))

    plt.ylabel(r'\mid F_D(q; \Delta{t})\mid^2', fontsize=16)
    plt.xlabel(r'Time (s)', fontsize=16)
    # plt.ylim(2E8, 5E8)
    # plt.xlim(0.1, 120)
    plt.legend(loc=1, bbox_to_anchor=(1, 0.23))
    # plt.savefig("wavevectors.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_image(ddm_dataset):
    """
    Plot the magnitude of the Fourier difference (colour intensity) as a function of
    time (y-axis) and wavevector (x-axis).
    """
    plt.imshow(np.log(ddm_dataset.data[1:]))
    # plt.savefig("ftOndDSlices.png", dpi=300)
    plt.show()


def plot_azimuthal_angle(data_series, frame_number, q_in_pixels=True):
    # Plot Fourier intensity as a function of wavevector for number of datasets at a specific frame number.

    for i in range(0, data_series.length):
        ddm_dataset = data_series.datasets[i]
        inverse_diameter = 1/ddm_dataset.pixels_per_particle
        # Plot the series in terms of q (1/d)
        if not q_in_pixels:
            plt.semilogy(np.arange(inverse_diameter, (inverse_diameter*(ddm_dataset.data[frame_number, :].shape[0])) + inverse_diameter, inverse_diameter), ddm_dataset.data[frame_number, :])
        # Plot the series in terms of q (pixels)
        else:
            plt.semilogy(np.arange(1, (ddm_dataset.data[frame_number, :].shape[0]) + 1, 1), ddm_dataset.data[frame_number, :])

    plt.ylabel(r'\mid F_D(q; \Delta{t})\mid^2', fontsize=16)
    if q_in_pixels:
        plt.xlabel(r'q (pixels)', fontsize=16)
    else:
        plt.xlabel(r'q (1/\sigma)', fontsize=16)
    plt.xlim(0, 200)
    # plt.ylim(5E6, 1E9)
    # plt.savefig("radialprofile.png", dpi=300)
    plt.show()

def save_series(data_series, output_limits):

    output_list = data_series.datasets[0].data[:, output_limits[0]:output_limits[1]]
    for i, wavevector in enumerate(range(*output_limits)):
        output_list[0, i] = wavevector

    np.savetxt("FTOneDSlices_cut.txt", output_list)