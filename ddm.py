import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from tempfile import TemporaryFile
import numba


def loadimages(analysis_radius, image_directory, file_prefix, file_suffix, octave_number, num_files, image_dimension):
    """Loads images, corrects for bleaching, performs fourier transforms and cuts images according to the analysis
    radius parameter. """
    print("Loading images.")
    # Calculate the maximum possible chunk size. A chunk always has a side length of power 2.
    chunk_size = int(2 ** (np.ceil(np.log2(np.min(image_dimension))) - octave_number + 1))
    # Analysis radius must be at most half the side length or analysis will fall outside the chunk
    if analysis_radius == 0 or analysis_radius > (chunk_size/2):
        analysis_radius = int(chunk_size/2)

    num_x_chunks = int(np.round(image_dimension[1] / chunk_size))
    num_y_chunks = int(np.round(image_dimension[0] / chunk_size))

    tmp_file = TemporaryFile()
    ftimagelist = np.memmap(tmp_file, mode='w+', dtype=np.complex128, shape=(num_files, num_x_chunks, num_y_chunks, analysis_radius*2, analysis_radius))

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
        for chunk_x in range(num_x_chunks):
            for chunk_y in range(num_y_chunks):
                image_chunk = get_chunk(tmp_array, chunk_x, chunk_y, chunk_size)
                # do the Fourier transform
                ft_tmp = np.fft.rfft2(image_chunk, s=(chunk_size, chunk_size))
                # Shift the quadrants so that low spatial frequencies are in the center of the 2D fourier transformed image.
                ft_tmp = np.fft.fftshift(ft_tmp, axes=(0,))
                # cut the image down to only include analysis_radius worth of pixels
                ft_tmp = ft_tmp[int(chunk_size/2 - analysis_radius):int(chunk_size/2 + analysis_radius), :analysis_radius]
                ftimagelist[file, chunk_x, chunk_y] = ft_tmp.copy()
    print("Image loading complete.")
    return ftimagelist, num_files, tmp_file, num_x_chunks, num_y_chunks


def setup_load_images(num_images, image_directory, file_prefix, file_suffix):
    if num_images == 0:
        file_list = glob(image_directory + file_prefix + "*" + file_suffix)
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    image_dimension = Image.open(image_directory + file_prefix + '0000' + file_suffix).size

    return num_files, image_dimension


def get_chunk(tmp_array, chunk_x, chunk_y, ft_size):
    # Dimensions of image [columns, rows]
    image_size = tmp_array.shape
    # Calculate the pixel indices bounding the chunk
    col_indices = [chunk_x * ft_size, chunk_x * ft_size + ft_size]
    if col_indices[1] > image_size[1]:
        col_indices[1] = image_size[1]
    row_indices = [chunk_y * ft_size, chunk_y * ft_size + ft_size]
    if row_indices[1] > image_size[0]:
        row_indices[1] = image_size[0]
    # Retrieve the chunk using the column and row indices
    image_chunk = tmp_array[row_indices[0]:row_indices[1], col_indices[0]:col_indices[1]]

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


def ddm_processing(binsize, cutoff, ftimagelist, numimages, num_octaves, num_x_chunks, num_y_chunks):
    """The ddm processing loop, this is passed a time series of any length of images of any size."""
    print("Processing octave " + str(num_octaves))
    r, nbins, histosamples, = initializeazimuthalaverage(ftimagelist[0, 0, 0], binsize)
    ftOneDSlices = np.zeros((numimages, num_x_chunks, num_y_chunks, nbins))
    samplecount = np.zeros(numimages)

    if cutoff > numimages:
        cutoff = numimages

    pbar = tqdm(total=int(((cutoff - 1) ** 2 + (cutoff - 1)) / 2 + (numimages - cutoff) * cutoff) * num_x_chunks * num_y_chunks)
    # Do the analysis
    for x_chunk in range(num_x_chunks):
        for y_chunk in range(num_x_chunks):
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
                    ftdiff = imagediff(ftimagelist[image1, x_chunk, y_chunk], ftimagelist[image2, x_chunk, y_chunk])
                    # Calculate the 2D power spectrum
                    ftdiff = twodpowerspectrum(ftdiff)
                    ftOneDSlices[framediff, x_chunk, y_chunk] += azimuthalaverage(r, ftdiff, histosamples)
                    if x_chunk + y_chunk == 0:  # Only need to count the frame differences once since it is the same each time
                        samplecount[framediff] += 1
                    loop_counter += 1
                pbar.update(loop_counter)
                loop_counter = 0
    pbar.close()

    # Normalise results, skipping the first empty row
    for x_chunk in range(num_x_chunks):
        for y_chunk in range(num_x_chunks):
            # normalise by the number of frames sampled
            ftOneDSlices[1:, x_chunk, y_chunk, :] = ftOneDSlices[1:, x_chunk, y_chunk, :] / np.transpose(np.array([samplecount[1:]]))
            # Normalise by the number of pixels in the chunk
            ftOneDSlices[1:, x_chunk, y_chunk, :] = ftOneDSlices[1:, x_chunk, y_chunk, :] / (ftimagelist[0, x_chunk, y_chunk].shape[0] * ftimagelist[0, x_chunk, y_chunk].shape[1])
            np.savetxt("results/FTOneDSlices_octave" + str(num_octaves) + "_x_" + '{:03d}'.format(x_chunk) + "_y_" + '{:03d}'.format(y_chunk) + ".txt", ftOneDSlices[:, x_chunk, y_chunk, :])
    print("Analysis Complete. Result saved to FTOneDSlices.txt")
    return 0


def main(binsize, cutoff, images_to_load, analysis_radius, image_directory, file_prefix, file_suffix, do_sub_analyses=False, max_octaves=5):

    if not do_sub_analyses:
        max_octaves = 1
    min_octaves = 1

    # The image setup needs to be done only once. This tells us how many images there are and what size they are.
    num_files, image_dimension = setup_load_images(images_to_load, image_directory, file_prefix, file_suffix)

    # Loop through the octaves. Each octave must reload the images as this is the stage where the FT is done.
    for octave in range(min_octaves, max_octaves + 1):
        ftimagelist, numimages, tmp_file, num_x_chunks, num_y_chunks = loadimages(analysis_radius, image_directory, file_prefix, file_suffix, octave, num_files, image_dimension)
        ddm_processing(binsize, cutoff, ftimagelist, numimages, octave, num_x_chunks, num_y_chunks)
        tmp_file.close()


main(binsize=0.25, cutoff=250, images_to_load=0, analysis_radius=100, image_directory="example_images/", file_prefix="iii_", file_suffix=".png", do_sub_analyses=True, max_octaves=4)
