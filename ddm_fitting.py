from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def fit_curves(timestep, n_data_points, octave_start=1, octave_end=1, debug=False):

    for num_octaves in range(octave_start, (octave_end+1)):
        for chunk in range(2 ** num_octaves):
            path = "results/FTOneDSlices_octave" + str(num_octaves) + "_part_" + '{:04d}'.format(chunk) + ".txt"
            data = np.loadtxt(path, skiprows=1)
            data = data[:n_data_points, :-1]  # assuming there are ~500 points, only keep the first 200 data points, cut the last column
            data_length = data.shape[0]

            first_third = int(data_length/3)
            second_third = int(data_length * 2 / 3)
            num_wavevectors = data.shape[1]-1

            xdata = np.arange(timestep, (data_length*timestep)+timestep, timestep)

            coefficients = np.zeros((data.shape[1], 3))

            for wavevector in range(num_wavevectors):
                if data[10, wavevector] != 0:
                    # Initial guess for a (plateau height), b (noise floor) and tau (relaxation time).
                    # These turn out to be fairly terrible estimates for certain curves but it seems to work reasonably.
                    a = np.mean(data[first_third:second_third, wavevector])
                    b = data[0, wavevector]
                    tau = timestep * 10
                    try:
                        params, covariance = curve_fit(exponential_fit, xdata[:second_third], data[:second_third, wavevector], [a, b, tau], bounds=([timestep/10, timestep/10, timestep/10], [1e15, 1e15, 1e15]))
                        coefficients[wavevector, :] = params
                        if debug:
                            yfunc = exponential_fit(xdata, *params)
                            plt.plot(xdata, yfunc, "-")
                            plt.figtext(0.2, 0.2, str(params[0]) + "  " + str(params[1]) + "  " + str(params[2]))
                    except ValueError:
                        pass
                        #print("Wavevector " + str(wavevector) + " not fitted.")
                    if debug:
                        yfunc = exponential_fit(xdata, a, b, tau)
                        plt.plot(xdata, yfunc, "-")
                        plt.plot(xdata, data[:, wavevector])
                        plt.figtext(0.2, 0.3, str(a) + "  " + str(b) + "  " + str(tau))
                        plt.show()

            #pixelsize = (np.round(2*np.pi/0.040, decimals=4))
            #wavevectors = np.arange(pixelsize, (data_length * pixelsize) + pixelsize, pixelsize)
            #plt.loglog(xdata[10:], coefficients[10:, 2], "o-")
            #plt.show()
            np.savetxt("results/fitting/" + str(num_octaves) + "_chunk_" + '{:04d}'.format(chunk) + "fittingparameters.txt", coefficients)
            print("Octave " + str(num_octaves) + ", chunk " + str(chunk) + " saved to fittingparameters.txt.")


def exponential_fit(x, a, b, tau):
    return a * (1-np.exp(-x/tau)) + b


fit_curves(float(1), 100, octave_start=1, octave_end=3)