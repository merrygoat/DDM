import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from tempfile import TemporaryFile
import numba


def loadimages(analysis_radius, image_directory, file_prefix, file_suffix, octave_number, num_files, min_size):
    """Loads images, corrects for bleaching, performs fourier transforms and cuts images according to the analysis
    radius parameter. """
    print("Loading images.")

    minimum_octave_size = 8     # minimum ocatave size in pixels

    if octave_number != 1:
        min_size = int(np.max((np.ceil(np.log2(min_size))) - octave_number - 1) ** 2)
        if min_size < minimum_octave_size:
            return 1

    if analysis_radius == 0 or analysis_radius > min_size:
        analysis_radius = int(min_size)/2

    tmp_file = TemporaryFile()
    ftimagelist = np.memmap(tmp_file, mode='w+', dtype=np.complex128, shape=(num_files, (octave_number ** 2), analysis_radius*2, analysis_radius))

    for file in range(num_files):
        tmp_image = Image.open(image_directory + file_prefix + '{:04d}'.format(file) + file_suffix)
        if tmp_image.mode == "RGB":
            tmp_image = tmp_image.convert(mode='L')
        tmp_array = np.array(tmp_image.copy())

        # Correct for bleaching by averaging the brightness across all images
        if file == 0:
            first_mean = np.mean(tmp_array)
        else:
            tmp_mean = np.mean(tmp_array)
            tmp_array = tmp_array * (first_mean / tmp_mean)
        
        for octave_segment in range((octave_number ** 2)):
            image_size = tmp_array.shape
            # calculate the nearest power of 2 to pad the FT array
            if octave_number == 1:
                ft_size = int(np.max(np.power(2, np.ceil(np.log2(image_size)))))
                tmp_octave = tmp_array
            else:
                ft_size = min_size
                octave_column = octave_segment % octave_number
                col_indices = [octave_column * min_size, octave_column * min_size + min_size]
                octave_row = octave_segment // octave_number
                row_indices = [octave_row * min_size, octave_row * min_size + min_size]
                tmp_octave = tmp_array[col_indices[0]:col_indices[1], row_indices[0]:row_indices[1]]

            # do the Fourier transform
            ft_tmp = (np.fft.rfft2(tmp_octave, s=(ft_size, ft_size)))
            # Shift the quadrants so that low spatial frequencies are in the center of the 2D fourier transformed image.
            ft_tmp = np.fft.fftshift(ft_tmp, axes=(0,))
            # cut the image down to only include analysis_radius worth of pixels
            if analysis_radius != 0:
                ft_tmp = ft_tmp[int(ft_size/2 - analysis_radius):int(ft_size/2 + analysis_radius), :analysis_radius]
            ftimagelist[file, octave_segment] = ft_tmp.copy()

    return ftimagelist, num_files, tmp_file


def setup_load_images(num_images, image_directory, file_prefix, file_suffix):
    if num_images == 0:
        file_list = glob(image_directory + file_prefix + "*" + file_suffix)
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    min_dimension = np.min(Image.open(image_directory + file_prefix + '0000' + file_suffix).size)

    return num_files, min_dimension


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


def ddm_processing(binsize, cutoff, ftimagelist, numimages, num_octaves):
    """The ddm processing loop, this is passed a time series of any length of images of any size."""

    r, nbins, histosamples, = initializeazimuthalaverage(ftimagelist[0, 0], binsize)
    ftOneDSlices = np.zeros((numimages, num_octaves, nbins))
    samplecount = np.zeros((numimages, num_octaves))

    if cutoff > numimages:
        cutoff = numimages

    # Do the analysis
    for octave in range(num_octaves):
        loop_counter = 0
        pbar = tqdm(total=int(((cutoff - 1) ** 2 + (cutoff - 1)) / 2 + (numimages - cutoff) * cutoff))
        for framediff in range(1, numimages):
            potential_frames = numimages - framediff
            if (numimages-framediff) < cutoff:
                frame_counter_max = potential_frames
            else:
                frame_counter_max = cutoff
            for frame_counter in range(0, frame_counter_max):
                image1 = int(potential_frames*frame_counter/frame_counter_max)
                image2 = image1 + framediff
                ftdiff = imagediff(ftimagelist[image1, octave], ftimagelist[image2, octave])
                # Calculate the 2D power spectrum
                ftdiff = twodpowerspectrum(ftdiff)
                ftOneDSlices[framediff, octave] += azimuthalaverage(r, ftdiff, histosamples)
                samplecount[framediff, octave] += 1
                loop_counter += 1
            pbar.update(loop_counter)
            loop_counter = 0
        pbar.close()

    # Normalise results, skipping the first empty row
    for i in range(1, numimages):
        ftOneDSlices[i] = ftOneDSlices[i] / samplecount[i]
    ftOneDSlices = ftOneDSlices / (ftimagelist[0, 0].shape[0] * ftimagelist[0, 0].shape[1])
    print("Analysis Complete. Result saved to FTOneDSlices.txt")
    for i in range(num_octaves):
        np.savetxt("FTOneDSlices_octave" + str(num_octaves) + "_part_" + '{:04d}'.format(i) + ".txt", ftOneDSlices[:, i])

    return 0


def main(binsize, cutoff, images_to_load, analysis_radius, image_directory, file_prefix, file_suffix, do_sub_analyses=False, max_octaves=5):

    if not do_sub_analyses:
        max_octaves = 1

    # The image setup needs to be done only once. This tells us how many images there are and what size they are.
    num_files, min_size = setup_load_images(images_to_load, image_directory, file_prefix, file_suffix)

    # Loop through the octaves. Each octave must reload the images as this is the stage where the FT is done.
    for octave in range(1, max_octaves + 1):
        ftimagelist, numimages, tmp_file = loadimages(analysis_radius, image_directory, file_prefix, file_suffix, octave, num_files, min_size)
        ddm_processing(binsize, cutoff, ftimagelist, numimages, octave)
        tmp_file.close()
