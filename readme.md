# Dynamic Digital Microscopy analysis code

This algorithm is based upon "Differential Dynamic Microscopy: Probing Wave Vector Dependent Dynamics with a Microscope", 
Cerbino, R and Trappe, V, PRL **100** 188102 (2008).

It is designed primarily for analysing dynamics of colloidal suspensions which do not have a high enough signal-noise ratio for conventional particle tracking. It should work with either confocal or brightfield microscopy images. The timestep separation should preferably be less than 10 particle relaxation times in order to get good results at the wavevector corresponding to particle diameter. Longer time series result in better statistics.

Example images can be found in the example folder. The "example_usage" Ipython notebook demonstrates basic usage of the analysis and plotting code.

Variables which have to be passed to the analysis code are:
* binsize         - Bin size for the histogram used in the radial averaging of the Fourier transform (FT)
* analysisradius  -	Radius of FT radial averaging. Set to 0 for analysis of full FT
* cutoff          -	Maximum averaging for each timestep. Set to 0 for analysis of all images.
* images_to_load  -	Number of timesteps to load from disk. Set to 0 for all available images.
* image_directory -	Directory path of images to analyse, end with slash
* file_prefix     - Prefix for file. Program assumes files are numbered sequentially from 0 with 4 digits i.e. image_0000.png, image_0001.png...
* file_suffix     - Image file type, ".png" or ".tif" are know to work. Other formats untested.

Required libraries: numpy, matplotlib, numba, tqdm, pillow
For the example file: IPython Notebook