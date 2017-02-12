import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)


def main():
    list_of_files = ["FTOneDSlices.txt"]
    pixel_size = [20]
    particle_size = [380]
    timestep = [0.526]

    data = Series(list_of_files, pixel_size, particle_size, timestep)

    plot_wavevector_range(data.datasets[0])
    # plot_wavevector_samples(data)
    # plot_image(data.datasets[0])
    # plot_azimuthal_angle(data)


class Series:
    """A class which contains one or more dataset objects"""
    datasets = []

    def __init__(self, file_list, pixel_list, particle_size_list, timestep_list):
        for i in range(len(file_list)):
            self.datasets.append(DDMDataset(file_list[i], pixel_list[i], particle_size_list[i], timestep_list[i]))
        self.length = len(self.datasets)


class DDMDataset:
    """A class which contains a single DDM dataset. The class has properties which describe the dataset"""

    def __init__(self, filename, pixel_size, particle_size, timestep):
        self.data = np.loadtxt(filename)
        self.pixel_size = np.array(pixel_size)
        self.particle_size = np.array(particle_size)
        self.pixels_per_particle = self.particle_size / self.pixel_size
        self.timestep = np.array(timestep)


def plot_wavevector_range(ddm_dataset):
    """Accepts a dataset object as input. Plots magnitude of Fourier difference at a range of different wavevectors for a single sample."""

    minwavevector = 10
    maxwavevector = 15
    wavevectorstep = 1

    for i in range(minwavevector, maxwavevector, wavevectorstep):
        wavevector = '{:1.2f}'.format(i / ddm_dataset.pixels_per_particle)
        plt.semilogx([j * ddm_dataset.timestep for j in range(ddm_dataset.data.shape[0])], ddm_dataset.data[:, i], label=wavevector)

    plt.ylabel(r'', fontsize=16)
    plt.xlabel(r'Time (s)', fontsize=16)
    # plt.ylim(3E8, 7E8)
    # plt.xlim(1, 200)
    plt.legend(loc=1)
    plt.savefig("wavevectors.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_wavevector_samples(data_series):
    """Plots magnitude of Fourier difference at a single wavevector for multiple samples."""

    wavevector = 25

    for i in range(0, data_series.length):
        current_dataset = data_series.datasets[i]
        plt.semilogx([i * current_dataset.timestep for i in range(current_dataset.data.shape[0])],
                     current_dataset.data[:, wavevector], label=str(current_dataset.pixel_size) + " nm px, q="
                    + '{:1.2f}'.format(wavevector / current_dataset.pixels_per_particle))

    plt.ylabel(r'\mid F_D(q; \Delta{t})\mid^2', fontsize=16)
    plt.xlabel(r'Time (s)', fontsize=16)
    # plt.ylim(2E8, 5E8)
    # plt.xlim(0.1, 120)
    plt.legend(loc=1, bbox_to_anchor=(1, 0.23))
    plt.savefig("wavevectors.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_image(ddm_dataset):
    """
    Plot the magnitude of the Fourier difference (colour intensity) as a function of
    time (y-axis) and wavevector (x-axis).
    """
    plt.imshow(np.log(ddm_dataset.data[1:]))
    plt.savefig("ftOndDSlices.png", dpi=300)
    plt.show()


def plot_azimuthal_angle(data_series):
    # Plot Fourier intensity as a function of wavevector for number of datasets at a specific frame number.

    time = 40
    # Plot with q in units of pixels. If not plot q in units of 1/d.
    q_in_pixels = True

    for i in range(0, data_series.length):
        ddm_dataset = data_series.datasets[i]
        inverse_diameter = 1/ddm_dataset.pixels_per_particle
        # Plot the series in terms of q (1/d)
        if not q_in_pixels:
            plt.semilogy(np.arange(inverse_diameter, (inverse_diameter*(ddm_dataset.data[time, :].shape[0])) + inverse_diameter, inverse_diameter), ddm_dataset.data[time, :])
        # Plot the series in terms of q (pixels)
        else:
            plt.semilogy(np.arange(1, (ddm_dataset.data[time, :].shape[0]) + 1, 1), ddm_dataset.data[time, :])

    plt.ylabel(r'\mid F_D(q; \Delta{t})\mid^2', fontsize=16)
    if q_in_pixels:
        plt.xlabel(r'q (pixels)', fontsize=16)
    else:
        plt.xlabel(r'q (1/\sigma)', fontsize=16)
    # plt.xlim(0, 10)
    plt.ylim(10000000, 10000000000)
    plt.savefig("radialprofile.png", dpi=300)
    plt.show()


main()
