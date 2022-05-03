import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import scipy.optimize
import statistics as st
from time import time
import os

#Constants related to the signal
fs = 10**6          #sampling rate
T = 1 / fs          #period
f0 = 10**5          #true frequency
w0 = 2*np.pi*f0     #true angular frequncy
phi = np.pi/8       #true phase
A = 1               #amplitude

#Variables related to the FFT
k = 20              #variable to adjust size of FFT
M = 2**k            #FFT size
N = 513             #lenght of signal
n0 = -256           #starting point

iterations=100      #Number of iterations to run code
step = 2 #Number of steps between each SNR value. Set to 10 to only run k=10 and 20. Set to 2 to run k=10 til 20
write_to_file_bool = True #If True, the results of the estimations are written to a CSV file

#time vector
t = np.linspace(n0*T, (n0+N-1)*T, N)

def generateSignal(sigma):
    #noise componentes
    real_noise = np.random.normal(0,sigma,N)
    complex_noise = np.random.normal(0,sigma,N)

    #signal with noise
    x =  A*np.exp(1j*(w0*t + phi)) + (real_noise + 1j*complex_noise)
    return x

def FFT(signal):
    yf = fft(signal, M)             #compute FFT of signal
    freq = fftfreq(M, 1/(fs))       #frequencies in the FFT

    #shift the zero frequency to be at the center of the spectrum
    yfs = fftshift(yf)              #shifted FFT of signal
    freqs = fftshift(freq)          #shifted frequencies
    return yf, freq, yfs, freqs

def calculateMLE(yf):
    arg_max = np.argmax(np.abs(yf))
    w_e = (2*np.pi*arg_max)/(M*T)                       #estimated omega
    phi_e = np.angle(yf[arg_max]*np.exp(-1j*w_e*n0*T))  #estimated phi

    return w_e, phi_e

def calculateErrors(w_e, phi_e):
    e_w = w0 - w_e
    e_phi = phi - phi_e
    
    return e_w, e_phi

def calculateEstimations(w_array, phi_array):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEstimate")
    var_w_e = np.var(w_array, ddof=1)
    var_phi_e = np.var(phi_array, ddof=1)
    print("Variance of estimated omega:\t\t", var_w_e)
    print("Variance of estimated phi:\t\t", var_phi_e)
    
    return var_w_e, var_phi_e

def calculateCRLB(sigma2):
    P = N*(N-1)/2
    Q = N*(N-1)*(2*N-1)/6

    var_omega = (12*sigma2)/((A**2)*(T**2)*N*((N**2)-1))
    var_phi = (12*sigma2)*((n0**2)*N+2*n0*P+Q)/((A**2)*(N**2)*((N**2)-1))
    
    return var_omega, var_phi

def plot(freqs, yfs):
    plt.plot(2*np.pi*freqs, 20*np.log10(np.abs(yfs)))
    plt.xlabel("Angular frequency [$\omega$]")
    plt.ylabel("Amplitude [dB]")
    plt.show()

def calculateSNR(yfs):
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nSNR\n")
    yfs_db = 20*np.log10(np.abs(yfs))
    signal_power = max(np.abs(yfs_db))
    noise_power = (sum(np.abs(yfs_db))-signal_power) / (len(yfs-1))
    SNR = signal_power - noise_power
    print(f"Signal power: {signal_power}")
    print(f"Noise power: {noise_power}")
    print(f"SNR: {SNR}")

def calculateMSE(w0,yf):
    w_var = w0[0]                                                     #guessed frequency
    s =  A*np.exp(1j*(w_var*t + phi))                                 #signal without noise
    yf_guss, freq_guess, yfs_guss, freqs_guess = FFT(s)               #fft of signal without noise
    
    mse = (np.square(np.abs(yf_guss) - np.abs(yf))).mean(axis=None)   #mean square error of the two fft's
    return mse

def nelderMeadSignalGenerator(sigma):
    x = generateSignal(sigma)
    yf, freq, yfs, freqs = FFT(x)
    arg_max = np.argmax(np.abs(yf))
    return yf, arg_max

def nelderMead(sigma, iterations):
    
    NMfreq = []
    NMfreqErrors = []
    
    time_before = time()
    
    for i in range(iterations):
        yf, arg_max = nelderMeadSignalGenerator(sigma)                       #generate signal and return fft
        
        #minimize mse using Nelder Mead method
        result  = scipy.optimize.minimize(calculateMSE, w0, args=yf, method = "Nelder-Mead")
        NM_w_estimate = result.x[0]                                     #frequency estimate
        
        NMfreq.append(NM_w_estimate)                                    #add estimate to list
        NMfreqErrors.append(w0 - result.x[0])                           #add frequency error to list
    
    time_after = time()
    time_dif = time_after - time_before
    
    #Calculate mean and variance of the lists   
    NM_freq_mean = st.mean(NMfreq)
    NM_freq_meanerror = st.mean(NMfreqErrors)
    NM_freq_var = st.variance(NMfreq)
    
    return NM_freq_mean, NM_freq_var, time_dif


def add_fileheader(filename):
    f = open(filename, "w")
    f.write("Sep=,\n")
    f.write("SNR,w,phi,w_var_estiamte,phi_var_estiamte,w_error,phi_error,w_error_var,phi_error_var,w_CRLB,phi_CRLB\n")
    f.close()

def write_to_file(filename, SNR, w, phi, w_var, phi_var, w_error, phi_error, w_error_var, phi_error_var, w_CRLB, phi_CRLB):
    f = open(filename, "a")
    f.write(str(SNR)+","+str(w)+","+str(phi)+","+str(w_var)+","+str(phi_var)+","+str(w_error)+","+str(phi_error)+","+str(w_error_var)+","+str(phi_error_var)+","+str(w_CRLB)+","+str(phi_CRLB)+"\n")
    f.close()

def add_fileheader_nelder(filename):
    f = open(filename, "w")
    f.write("Sep=,\n")
    f.write("SNR,w_NM,w_var_NM,w_20,w_var_20,w_10,w_var_10,w_CRLB\n")
    f.close()

def write_to_file_nelder(filename, SNR, w_NM, w_var_NM, w_20, w_var_20, w_10, w_var_10, w_CRLB):
    f = open(filename, "a")
    f.write(str(SNR)+","+str(w_NM)+","+str(w_var_NM)+","+str(w_20)+","+str(w_var_20)+","+str(w_10)+","+str(w_var_10)+","+str(w_CRLB)+"\n")
    f.close()

def main_estimate():
    time_array_10 = []
    time_array_20 = []
    NM_w_array_10 = []
    NM_w_array_20 = []
    NM_w_var_array_10 = []
    NM_w_var_array_20 = []
    os.mkdir("results")
    
    global k            #Use global to acces global variable k
    k = 10              
    
    for k in range(10,22,step):
        global M            #Use global to acces global variable M
        M = 2**k
        print(f"\nRunning with size {k}")
        filename = f"results/estiamte_{M}.csv"
        if write_to_file_bool:
            add_fileheader(filename)
        
        for SNR_DB in  range (-10, 70, 10):
            SNR_A = 10**((SNR_DB)/10)           #SNR in voltage ratio
            sigma2 = (A**2)/(2*SNR_A)           #variance of the noise
            sigma = np.sqrt(sigma2)             #standard deviation of the noise
            print(f"###################SNR: {SNR_DB} ###############################")
            
            w_array = []
            phi_array = []
            w_err_array = []
            phi_err_array = []
            
            time_before = time() #time before 100 iterations
            for i in range(0, iterations):
                #print(f"\n##########################################################\nNumber {i}\n")
                x = generateSignal(sigma)
                yf, freq, yfs, freqs = FFT(x)
                
                w_e, phi_e = calculateMLE(yf)
                w_array.append(w_e)
                phi_array.append(phi_e)
                
                err_w, err_phi = calculateErrors(w_e, phi_e)
                w_err_array.append(err_w)
                phi_err_array.append(err_phi)
            
            time_after = time() #time after 100 iterations
            var_w_e, var_phi_e = calculateEstimations(w_array, phi_array)
            w_CRLB, phi_CRLB = calculateCRLB(sigma2)
            
            mean_w = np.mean(w_array)
            mean_phi = np.mean(phi_array)
            mean_err_w = np.mean(w_err_array)
            mean_err_phi = np.mean(phi_err_array)
            var_err_w = np.var(w_err_array, ddof=1)
            var_err_phi = np.var(phi_err_array, ddof=1)
            
            if k == 20:
                time_array_20.append(time_after - time_before)
                NM_w_array_20.append(mean_w)
                NM_w_var_array_20.append(var_w_e)
            elif k == 10:
                time_array_10.append(time_after - time_before)
                NM_w_array_10.append(mean_w)
                NM_w_var_array_10.append(var_w_e)
            
            if write_to_file_bool:
                write_to_file(filename, SNR_DB, mean_w, mean_phi, var_w_e, var_phi_e, mean_err_w, mean_err_phi, var_err_w, var_err_phi, w_CRLB, phi_CRLB)
        
        print()
        print(f"Results with M=2^{k}={M}")
        print("Angular frequency at SNR=60dB:", mean_w)
        print("Angular frequency error at SNR=60dB:", mean_err_w)
        print("Angular frequency variance at SNR=60dB:", var_err_w)
        print()
        print("Phase at SNR=60dB:", mean_phi)
        print("Phase error at SNR=60dB:", mean_err_phi)
        print("Phase variance at SNR=60dB:", var_err_phi)    

    return time_array_20, time_array_10, NM_w_array_20, NM_w_var_array_20, NM_w_array_10, NM_w_var_array_10


def main_NM(time_array_20, time_array_10, mean_w_20, var_w_20, mean_w_10, var_w_10):
    filename = "results/estiamte_nelder.csv"
    if write_to_file_bool:
        add_fileheader_nelder(filename)
    
    NM_time_array = []
    NM_freq_mean_array = []
    NM_freq_meanerror = []
    NM_freq_var = []
    
    i = 0
    global k            #Use global to acces global variable k
    k = 10              
    global M            #Use global to acces global variable M
    M = 2**k
    
    print("\nUsing Nelder Mead")
    for SNR_DB in range (-10, 70, 10):
        SNR_A = 10**((SNR_DB)/10)           #SNR in voltage ratio
        sigma2 = (A**2)/(2*SNR_A)           #variance of the noise
        sigma = np.sqrt(sigma2)             #standard deviation of the noise
        print(f"###################SNR: {SNR_DB} ###############################")
    
        w_NM, w_NM_var, time_NM = nelderMead(sigma, iterations)
        NM_time_array.append(time_NM)
        NM_freq_mean_array.append(w_NM)
        NM_freq_meanerror.append(w0 - w_NM)
        NM_freq_var.append(w_NM_var)
        
        w_CRLB,_ = calculateCRLB(sigma2)

        if write_to_file_bool:
            write_to_file_nelder(filename, SNR_DB, w_NM, w_NM_var, mean_w_20[i], var_w_20[i], mean_w_10[i], var_w_10[i], w_CRLB)
        i+=1
    
    print("Angular frequency using Nelder Mead for SNR=60dB:", NM_freq_mean_array[len(NM_freq_mean_array)-1])
    print("Angular frequency error using Nelder Mead for SNR=60dB:", NM_freq_meanerror[len(NM_freq_meanerror)-1])
    print("Angular Frequency variance using Nelder Mead for SNR=60dB:", NM_freq_var[len(NM_freq_meanerror)-1])
    print()
    
    print(f"Average time using normal method (M=2^20)[s]:", np.mean(time_array_20))
    print(f"Average time using normal method (M=2^10)[s]:", np.mean(time_array_10))
    print("Average time using Nelder Mead (M=2^10)[s]:", np.mean(NM_time_array))

time_array_20, time_array_10, mean_w_20, var_w_20, mean_w_10, var_w_10 = main_estimate()
main_NM(time_array_20, time_array_10, mean_w_20, var_w_20, mean_w_10, var_w_10)
    
