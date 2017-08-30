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

    # Calculate the maximum possible chunk size. A chunk always has a side length of power 2.
    chunk_size = int(2 ** (np.ceil(np.log2(min_size)) - octave_number + 1))
    # Analysis radius must be at most half the side length or it will fall off the chunk
    if analysis_radius == 0 or analysis_radius > (chunk_size/2):
        analysis_radius = int(chunk_size/2)

    tmp_file = TemporaryFile()
    ftimagelist = np.memmap(tmp_file, mode='w+', dtype=np.complex128, shape=(num_files, (2 ** octave_number), analysis_radius*2, analysis_radius))

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

        # Get chunks of the image and calculate their Fourier transforms
        for octave_segment in range((2 ** octave_number)):
            image_chunk = get_chunk(tmp_array, octave_number, octave_segment, chunk_size)
            # do the Fourier transform
            ft_tmp = (np.fft.rfft2(image_chunk, s=(chunk_size, chunk_size)))
            # Shift the quadrants so that low spatial frequencies are in the center of the 2D fourier transformed image.
            ft_tmp = np.fft.fftshift(ft_tmp, axes=(0,))
            # cut the image down to only include analysis_radius worth of pixels
            if analysis_radius != 0:
                ft_tmp = ft_tmp[int(chunk_size/2 - analysis_radius):int(chunk_size/2 + analysis_radius), :analysis_radius]
            ftimagelist[file, octave_segment] = ft_tmp.copy()
    print("Image loading complete.")
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


def get_chunk(tmp_array, octave_number, octave_segment, ft_size):
    image_size = tmp_array.shape
    # calculate the nearest power of 2 to pad the FT array
    if octave_number == 1:
        image_chunk = tmp_array
    else:
        # Calculate the column of the chunk and the pixel indices bounding it
        octave_column = octave_segment % octave_number
        if octave_number - octave_column == 1:
            col_indices = [octave_column * ft_size, image_size[1]]
        else:
            col_indices = [octave_column * ft_size, octave_column * ft_size + ft_size]
        # Calculate the row of the chunk and the pixel indices bounding it
        octave_row = octave_segment // octave_number
        if octave_number - octave_row == 1:
            row_indices = [octave_row * ft_size, image_size[0]]
        else:
            row_indices = [octave_row * ft_size, octave_row * ft_size + ft_size]
        # Retrieve the chunk using the column and row indices
        image_chunk = tmp_array[col_indices[0]:col_indices[1], row_indices[0]:row_indices[1]]

    return image_chunk


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
    print("Processing octave " + str(num_octaves))
    r, nbins, histosamples, = initializeazimuthalaverage(ftimagelist[0, 0], binsize)
    ftOneDSlices = np.zeros((numimages, 2 ** num_octaves, nbins))
    samplecount = np.zeros(numimages)

    if cutoff > numimages:
        cutoff = numimages

    pbar = tqdm(total=int(((cutoff - 1) ** 2 + (cutoff - 1)) / 2 + (numimages - cutoff) * cutoff) * 2 ** num_octaves)
    # Do the analysis
    for octave in range(2 ** num_octaves):
        loop_counter = 0
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
                if octave == 0:
                    samplecount[framediff] += 1
                loop_counter += 1
            pbar.update(loop_counter)
            loop_counter = 0
    pbar.close()

    # Normalise results, skipping the first empty row
    for octave in range(2 ** num_octaves):
        ftOneDSlices[1:, octave, :] = ftOneDSlices[1:, octave, :] / np.transpose(np.array([samplecount[1:]]))
    ftOneDSlices = ftOneDSlices / (ftimagelist[0, 0].shape[0] * ftimagelist[0, 0].shape[1])
    print("Analysis Complete. Result saved to FTOneDSlices.txt")
    for i in range(2 ** num_octaves):
        np.savetxt("results/FTOneDSlices_octave" + str(num_octaves) + "_part_" + '{:04d}'.format(i) + ".txt", ftOneDSlices[:, i, :])

    return 0


def main(binsize, cutoff, images_to_load, analysis_radius, image_directory, file_prefix, file_suffix, do_sub_analyses=False, max_octaves=5):

    if not do_sub_analyses:
        max_octaves = 1
    min_octaves = 1

    # The image setup needs to be done only once. This tells us how many images there are and what size they are.
    num_files, min_size = setup_load_images(images_to_load, image_directory, file_prefix, file_suffix)

    # Loop through the octaves. Each octave must reload the images as this is the stage where the FT is done.
    for octave in range(min_octaves, max_octaves + 1):
        ftimagelist, numimages, tmp_file = loadimages(analysis_radius, image_directory, file_prefix, file_suffix, octave, num_files, min_size)
        ddm_processing(binsize, cutoff, ftimagelist, numimages, octave)
        tmp_file.close()


main(binsize=0.25, cutoff=250, images_to_load=0, analysis_radius=100, image_directory="example_images/", file_prefix="iii_", file_suffix=".png", do_sub_analyses=True, max_octaves=6)
