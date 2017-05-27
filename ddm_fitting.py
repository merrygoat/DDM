from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit

def fit_curves():
    
    if len(argv) != 3:
        print("Incorrect arguments, use ./ddm_fitting.py file_to_process timestep")
        exit()
    path = argv[1]
    timestep = float(argv[2])
    data = np.loadtxt(path, skiprows=1)
    data = data[:200, :-1]  # assuming there are ~500 points, only keep the first 200 data points, cut the last column
    data_length = data.shape[0]
    
    first_third = int(data_length/3)
    second_third = int(data_length * 2 / 3)
    num_wavevectors = data.shape[1]-1

    xdata = np.arange(timestep, (data_length*timestep)+1, timestep)

    coefficients = np.zeros((data.shape[1], 3))
    
    for wavevector in range(num_wavevectors):
        if data[10, wavevector] != 0:
            # Initial guess for a (plateau height), b (noise floor) and tau (relaxation time)
            initial_params =[np.mean(data[first_third:second_third, wavevector]), data[0, wavevector], xdata[10]]
            params, covariance = curve_fit(exponential_fit, xdata[:second_third], data[:second_third, wavevector], initial_params, bounds=([0.1, 0.1, 0.1], [1e10, 1e10, 1e10]))
            #plt.plot(xdata, data[:, wavevector])
            #yfunc = exponential_fit(xdata, *params)
            #plt.plot(xdata, yfunc, "-")
            #plt.show()
            coefficients[wavevector, :] = params
            

    #pixelsize = (np.round(2*np.pi/0.040, decimals=4))
    #wavevectors = np.arange(pixelsize, (data_length * pixelsize) + pixelsize, pixelsize)
    #plt.loglog(xdata[10:], coefficients[10:, 2], "o-")
    #plt.show()
    np.savetxt("fittingparameters.txt", coefficients)

def exponential_fit(x, a, b, tau):
    return a * (1-np.exp(-x/tau)) + b

fit_curves()
