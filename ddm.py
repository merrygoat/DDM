import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from sys import argv
import numba


def loadimages(num_images, analysis_radius, image_directory, file_prefix, file_suffix):
    """Loads images, corrects for bleaching, performs fourier transforms and cuts images according to the analysis
    radius parameter. """
    print("Loading images.")
    ftimagelist = []

    cut_image_shape, num_files = setup_load_images(num_images, image_directory, file_prefix, file_suffix, analysis_radius)

    for i in range(num_files):
        tmp_image = Image.open(image_directory + file_prefix + '{:04d}'.format(i) + file_suffix)
        tmp_array = np.array(tmp_image.copy(), dtype=np.int16)

        image_size = tmp_array.shape
        # calculate the nearest power of 2 to pad the FT array
        ft_size = int(np.max(np.power(2, np.ceil(np.log2(image_size)))))

        # Correct for bleaching by averaging the brightness across all images
        if i == 0:
            first_mean = np.mean(tmp_array)
        else:
            tmp_mean = np.mean(tmp_array)
            tmp_array = tmp_array * (first_mean / tmp_mean)

        # do the Fourier transform
        ft_tmp = (np.fft.fft2(tmp_array, s=(ft_size, ft_size)))
        # Shift the quadrants so that low spatial frequencies are in the center of the 2D fourier transformed image.
        ft_tmp = np.fft.fftshift(ft_tmp)
        # cut the image down to only include analysis_radius worth of pixels
        ftimagelist.append(ft_tmp[cut_image_shape[0]:cut_image_shape[1], cut_image_shape[2]:cut_image_shape[3]].copy())

    print("Image Loading complete. Beginning analysis.")
    return ftimagelist, num_files


def setup_load_images(num_images, image_directory, file_prefix, file_suffix, analysis_radius):
    if num_images == 0:
        file_list = glob(image_directory + "*" + file_suffix)
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    # test load an image to get size
    tmp_image = Image.open(image_directory + file_prefix + "0000" + file_suffix)
    original_image_shape = tmp_image.size

    # determine what size to cut the image to
    if analysis_radius == 0:
        cut_image_shape = [0, original_image_shape[0], 0, original_image_shape[1]]  # xmin, xmax, ymin, ymax
    elif analysis_radius * 2 > np.min(original_image_shape):
        print("Error: analysisradius is larger than one of the image dimensions.")
        raise KeyboardInterrupt
    else:
        cut_image_shape = [original_image_shape[0] / 2 - analysis_radius, original_image_shape[0] / 2 + analysis_radius,
                           original_image_shape[1] / 2 - analysis_radius, original_image_shape[1] / 2 + analysis_radius]

    cut_image_shape = [int(i) for i in cut_image_shape]  # Make all values in cut_image_shape integers

    return cut_image_shape, num_files


def initializeazimuthalaverage(image, binsize):
    y, x = np.indices(image.shape)
    center = np.array([(image.shape[0] - 1) / 2.0, (image.shape[0] - 1) / 2.0])
    r = np.array(np.hypot(x - center[0], y - center[1]) / binsize, dtype=np.int)
    maxbin = np.max(r)
    nbins = maxbin + 1
    bins = np.linspace(0, maxbin, nbins + 1)
    histosamples = np.histogram(r, bins, )[0]

    return r, nbins, histosamples


def azimuthalaverage(r, image, histosamples):
    return np.bincount(np.ravel(r), weights=np.ravel(image)) / histosamples


@numba.jit
def imagediff(image1, image2):
    return image1 - image2


@numba.jit
def twodpowerspectrum(image):
    return image.real ** 2 + image.imag ** 2


def main():
    """This function will be called from the command line."""

    if len(argv) != 8:
        print("Incorrect syntax. Use ./ddm.py binsize, analysis_radius, cutoff, images_to_load, image_directory file_prefix file_suffix.\n See Readme for more detials.")
        raise KeyboardInterrupt
    else:
        binsize = int(argv[1])
        analysisradius = int(argv[2])
        cutoff = int(argv[3])
        images_to_load = int(argv[4])
        image_directory = argv[5]
        file_prefix = argv[6]
        file_suffix = argv[7]
        ddm_processing(binsize, analysisradius, cutoff, images_to_load, image_directory, file_prefix, file_suffix)


def ddm_processing(binsize, analysisradius, cutoff, images_to_load, image_directory, file_prefix, file_suffix):
    """If calling functions from within python this is the main loop."""

    # Load the images
    ftimagelist, numimages = loadimages(images_to_load, analysisradius, image_directory, file_prefix, file_suffix)

    r, nbins, histosamples, = initializeazimuthalaverage(ftimagelist[0], binsize)

    ftOneDSlices = np.zeros((numimages, nbins))
    samplecount = np.zeros(numimages)

    if cutoff > numimages:
        cutoff = numimages

    loop_counter = 0
    pbar = tqdm(total=int(((cutoff - 1) ** 2 + (cutoff - 1)) / 2 + (numimages - cutoff) * cutoff))
    # Do the analysis
    for i in range(cutoff):
        for j in range(i + 1, numimages):
            ftdiff = imagediff(ftimagelist[i], ftimagelist[j])
            # Calculate the 2D power spectrum
            ftdiff = twodpowerspectrum(ftdiff)
            ftOneDSlices[j - i] += azimuthalaverage(r, ftdiff, histosamples)

            samplecount[j - i] += 1
            loop_counter += 1
        pbar.update(loop_counter)
        loop_counter = 0

    # Normalise results, skipping the first empty row
    for i in range(1, numimages):
        ftOneDSlices[i] = ftOneDSlices[i] / samplecount[i]
    ftOneDSlices = ftOneDSlices / (ftimagelist[0].shape[0] * ftimagelist[0].shape[1])
    print("Analysis Complete. Result saved to FTOneDSlices.txt")
    np.savetxt("FTOneDSlices.txt", ftOneDSlices)

main()