import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from tempfile import TemporaryFile
import numba


def loadimages(num_images, analysis_radius, image_directory, file_prefix, file_suffix, num_octaves):
    """Loads images, corrects for bleaching, performs fourier transforms and cuts images according to the analysis
    radius parameter. """
    print("Loading images.")

    minimum_octave_size = 8     # minimum ocatave size in pixels

    num_files, analysis_radius, min_dimension = setup_load_images(num_images, image_directory, file_prefix, file_suffix, analysis_radius)

    if num_octaves != 0:
        min_dimension = int(np.max((np.ceil(np.log2(min_dimension)))-num_octaves)**2)
        if min_dimension < minimum_octave_size:
            return 1

    if analysis_radius > min_dimension:
        analysis_radius = min_dimension

    tmp_file = TemporaryFile()
    ftimagelist = np.memmap(tmp_file, mode='w+', dtype=np.complex128, shape=((num_octaves+1**2), num_files, analysis_radius*2, analysis_radius))

    for file in range(num_files):
        tmp_image = Image.open(image_directory + file_prefix + '{:04d}'.format(file) + file_suffix)
        if tmp_image.mode == "RGB":
            tmp_image = tmp_image.convert(mode='L')
        tmp_array = np.array(tmp_image.copy())

        for octave_segment in range((num_octaves + 1 ** 2)):

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
            ft_tmp = (np.fft.rfft2(tmp_array, s=(ft_size, ft_size)))
            # Shift the quadrants so that low spatial frequencies are in the center of the 2D fourier transformed image.
            ft_tmp = np.fft.fftshift(ft_tmp, axes=(0,))
            # cut the image down to only include analysis_radius worth of pixels
            if analysis_radius != 0:
                ft_tmp = ft_tmp[int(ft_size/2 - analysis_radius):int(ft_size/2 + analysis_radius), :analysis_radius]
            ftimagelist[i] = ft_tmp.copy()

    return ftimagelist, num_files, tmp_file


def setup_load_images(num_images, image_directory, file_prefix, file_suffix, analysis_radius):
    if num_images == 0:
        file_list = glob(image_directory + file_prefix + "*" + file_suffix)
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    min_dimension = np.min(Image.open(image_directory + file_prefix + '0000' + file_suffix).size)

    if analysis_radius == 0 or analysis_radius > min_dimension:
        analysis_radius = int(min_dimension/2)

    return num_files, min_dimension, analysis_radius


def initializeazimuthalaverage(image, binsize):
    y, x = np.indices(image.shape)
    center = np.array([image.shape[0] / 2, 0])
    r = np.array(np.hypot(x - center[1], y - center[0]) / binsize, dtype=np.int)
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


def ddm_processing(binsize, cutoff, ftimagelist, numimages):
    """The ddm processing loop, this is passed a time series of any length of images of any size."""

    r, nbins, histosamples, = initializeazimuthalaverage(ftimagelist[0], binsize)
    ftOneDSlices = np.zeros((numimages, nbins))
    samplecount = np.zeros(numimages)

    if cutoff > numimages:
        cutoff = numimages

    loop_counter = 0
    pbar = tqdm(total=int(((cutoff - 1) ** 2 + (cutoff - 1)) / 2 + (numimages - cutoff) * cutoff))
    # Do the analysis
    for framediff in range(1, numimages):
        potential_frames = numimages - framediff
        if (numimages-framediff) < cutoff:
            frame_counter_max = potential_frames
        else:
            frame_counter_max = cutoff
        for frame_counter in range(0, frame_counter_max):
            image1 = int(potential_frames*frame_counter/frame_counter_max)
            image2 = image1 + framediff
            ftdiff = imagediff(ftimagelist[image1], ftimagelist[image2])
            # Calculate the 2D power spectrum
            ftdiff = twodpowerspectrum(ftdiff)
            ftOneDSlices[framediff] += azimuthalaverage(r, ftdiff, histosamples)
            samplecount[framediff] += 1
            loop_counter += 1
        pbar.update(loop_counter)
        loop_counter = 0
    pbar.close()

    # Normalise results, skipping the first empty row
    for i in range(1, numimages):
        ftOneDSlices[i] = ftOneDSlices[i] / samplecount[i]
    ftOneDSlices = ftOneDSlices / (ftimagelist[0].shape[0] * ftimagelist[0].shape[1])
    print("Analysis Complete. Result saved to FTOneDSlices.txt")
    np.savetxt("FTOneDSlices.txt", ftOneDSlices)

    return 0


def main(binsize, cutoff, images_to_load, analysisradius, image_directory, file_prefix, file_suffix, do_sub_analyses=False, max_octaves=5):

    if not do_sub_analyses:
        max_octaves = 1

    for octave in range(max_octaves):
        ftimagelist, numimages, tmp_file = loadimages(images_to_load, analysisradius, image_directory, file_prefix, file_suffix, octave)
        ddm_processing(binsize, cutoff, ftimagelist, numimages)
        tmp_file.close()