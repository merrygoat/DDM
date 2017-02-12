import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob


def loadimages(num_images, analysis_radius):
    print("Loading images.")
    ftimagelist = []
    image_directory = "E:\\Confocal\\STED\\Hard Spheres\\17-02-02\\RITC 23\\i\\images\\"
    file_prefix = "i_"

    cut_image_shape, num_files = setup_load_images(num_images, image_directory, file_prefix, analysis_radius)

    for i in range(num_files):
        tmp_image = Image.open(image_directory + file_prefix + '{:04d}'.format(i) + ".png")
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
        ftimagelist.append(np.fft.fft2(tmp_array, s=(ft_size, ft_size)))
        # Shift the quadrants so that low spatial frequencies are in the center of the 2D fourier transformed image.
        ftimagelist[i] = np.fft.fftshift(ftimagelist[i])
        # cut the image down to only include analysis_radius worth of pixels
        ftimagelist[i] = ftimagelist[i][cut_image_shape[0]:cut_image_shape[1], cut_image_shape[2]:cut_image_shape[3]]

    print("Image Loading complete. Beginning analysis.")
    return ftimagelist, num_files


def setup_load_images(num_images, image_directory, file_prefix, analysis_radius):
    if num_images == 0:
        file_list = glob(image_directory + "*.png")
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    # test load an image to get size
    tmp_image = Image.open(image_directory + file_prefix + "0000.png")
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
    r = np.hypot(x - center[0], y - center[1])
    maxbin = int(np.ceil(np.max(r)))
    nbins = int(np.ceil(maxbin / binsize))
    bins = np.linspace(0, maxbin, nbins + 1)
    histosamples = np.histogram(r, bins, )[0]

    return r, nbins, maxbin, histosamples


def azimuthalaverage(r, nbins, maxbin, image, histosamples):
    return np.histogram(r, nbins, range=(0, maxbin), weights=image)[0] / histosamples


def imagediff(image1, image2):
    return image1 - image2


def twodpowerspectrum(image):
    return np.abs(image) ** 2


def main():
    binsize = 1  # Bin size for the histogram used in the radial averaging of the Fourier transform (FT)
    analysisradius = 100  # Radius of FT radial averaging. Set to 0 for analysis of full FT
    cutoff = 250  # Maximum averaging for each timestep. Set to 0 for analysis of all images.
    images_to_load = 0  # Number of timesteps to load from disk. Set to 0 for all available images.

    # Load the images
    ftimagelist, numimages = loadimages(images_to_load, analysisradius)

    r, nbins, maxbin, histosamples,  = initializeazimuthalaverage(ftimagelist[0], binsize)

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

            ftOneDSlices[j - i] += np.abs(azimuthalaverage(r, nbins, maxbin, ftdiff, histosamples))
            samplecount[j - i] += 1
            loop_counter += 1
        pbar.update(loop_counter)
        loop_counter = 0

    # Normalise results, skipping the first empty row
    for i in range(1, numimages):
        ftOneDSlices[i] = ftOneDSlices[i] / samplecount[i]
    ftOneDSlices = ftOneDSlices / (ftimagelist[0].shape[0] * ftimagelist[0].shape[1])

    np.savetxt("FTOneDSlices.txt", ftOneDSlices)

main()
