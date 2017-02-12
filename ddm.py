import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob


def loadimages(num_images):
    ftimagelist = []
    image_directory = "E:\\Confocal\\STED\\Hard Spheres\\17-01-27\\RITC 18\\i\\images\\"
    file_prefix = "i_"

    if num_images == 0:
        file_list = glob(image_directory + "*.png")
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    for i in range(num_files):
        tmp_image = Image.open(image_directory + file_prefix + '{:04d}'.format(i) + ".png")
        tmp_array = np.array(tmp_image.copy(), dtype=np.int16)

        # Correct for bleaching by averaging the brightness across all images
        if i == 0:
            first_mean = np.mean(tmp_array)
        else:
            tmp_mean = np.mean(tmp_array)
            tmp_array = tmp_array * (first_mean / tmp_mean)

        ftimagelist.append(np.fft.fft2(tmp_array))
        # Shift the quadrants so that low spatial frequencies are in the center of the 2D fourier transformed image.
        ftimagelist[i] = np.fft.fftshift(ftimagelist[i])

    return ftimagelist, num_files


def initializeazimuthalaverage(image, binsize, analysisradius):
    original_image_shape = image.shape

    if analysisradius == 0:
        cut_image_shape = [0, original_image_shape[0], 0, original_image_shape[1]]   # xmin, xmax, ymin, ymax
    elif analysisradius * 2 > np.min(original_image_shape):
        print("Error: analysisradius is larger than one of the image dimensions.")
        raise KeyboardInterrupt
    else:
        # It is worthwhile taking the center of the image over the top left. Edges are more susceptable to aberration.
        cut_image_shape = [original_image_shape[0] / 2 - analysisradius, original_image_shape[0] / 2 + analysisradius,
                           original_image_shape[1] / 2 - analysisradius, original_image_shape[1] / 2 + analysisradius]
    cut_image_shape = [int(i) for i in cut_image_shape]   # Make all values in cut_image_shape integers
    image = image[cut_image_shape[0]:cut_image_shape[1], cut_image_shape[2]:cut_image_shape[3]]

    y, x = np.indices(image.shape)
    center = np.array([(image.shape[0] - 1) / 2.0, (image.shape[0] - 1) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    maxbin = int(np.ceil(np.max(r)))
    nbins = int(np.ceil(maxbin / binsize))
    bins = np.linspace(0, maxbin, nbins + 1)
    histosamples = np.histogram(r, bins, )[0]

    return r, nbins, maxbin, histosamples, cut_image_shape


def azimuthalaverage(r, nbins, maxbin, image, histosamples):
    return np.histogram(r, nbins, range=(0, maxbin), weights=image)[0] / histosamples


def imagediff(image1, image2, bounds):
    return image1[bounds[0]:bounds[1], bounds[2]:bounds[3]] - image2[bounds[0]:bounds[1], bounds[2]:bounds[3]]


def twodpowerspectrum(image):
    return np.abs(image) ** 2


def main():
    binsize = 1             # Bin size for the histogram used in the radial averaging of the Fourier transform (FT)
    analysisradius = 100    # Radius of FT radial averaging. Set to 0 for analysis of full FT
    cutoff = 250            # Maximum averaging for each timestep. Set to 0 for analysis of all images.
    images_to_load = 100    # Number of timesteps to load from disk. Set to 0 for all available images.

    # Load the images
    ftimagelist, numimages = loadimages(images_to_load)

    r, nbins, maxbin, histosamples, imageshape = initializeazimuthalaverage(ftimagelist[0], binsize, analysisradius)

    ftOneDSlices = np.zeros((numimages, nbins))
    samplecount = np.zeros(numimages)

    if cutoff > numimages:
        cutoff = numimages

    loop_counter = 0
    pbar = tqdm(total=int(((cutoff-1)**2+(cutoff-1))/2 + (images_to_load-cutoff)*cutoff))
    # Do the analysis
    for i in range(cutoff):
        for j in range(i + 1, numimages):
            ftdiff = imagediff(ftimagelist[i], ftimagelist[j], imageshape)
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