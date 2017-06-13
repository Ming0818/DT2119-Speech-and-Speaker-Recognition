# DT2118, Lab 1 Feature Extraction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as ssi
import scipy.fftpack as sfft
import tools
import scipy.cluster.hierarchy as hac
from sklearn.mixture import GaussianMixture as GMM
import itertools


# load data:
example = np.load('../Datasets/example_python3.npz')['example'].item()
tidigits = np.load('../Datasets/tidigits_python3.npz')['tidigits']

samples = example['samples']

# Function given by the exercise ----------------------------------

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return tools.lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    res = []
    for i in range(0, len(samples), winshift):
        if i+winlen <= len(samples):
            res.append(samples[i:i+winlen])
        else:
            break    
     
    return np.array(res)

def preemp(input, p):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    return ssi.lfilter([1, -p], [1], input)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    M = len(input[0])
    ham_win = ssi.hamming(M, sym = True)
    
    return input * ham_win

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    f = sfft.fft(input,nfft)
    # print("f =", f)
    # print("f.real =", f.real)
    # print("f.imag =", f.imag)

    return f.real ** 2 + f.imag ** 2        

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    global fbank
    nfft = len(input[0])
    fs = samplingrate
    fbank = tools.trfbank(fs, nfft)
    Melspec = np.dot(input, fbank.T)
    return np.log(Melspec)
     
def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return sfft.dct(input, norm = "ortho")[:, 0:nceps]

def dtw(loc_dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    d = np.zeros(loc_dist.shape)
    N, M = loc_dist.shape
    for i in range(N):
        for j in range(M):
            d[i, j] = min(d[i - 1, j], d[ i - 1, j - 1], d[i, j - 1]) + loc_dist[i, j]
            
    
    return d[-1, -1]      



def mspec(samples, winlen = 400, winshift = 200, nfft=512, nceps=13, samplingrate=20000):
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, 0.97)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def locdist(v1, v2):
    locdist = np.zeros([len(v1), len(v2)])
    for i in range(len(v1)):
        for j in range(len(v2)):
            locdist[i, j] = np.linalg.norm(v1[i] - v2[j])
    return locdist


def main():
    #------------------------------------------------ Quest 4 ----------------------------->>>
    # Step 1: Enframe
    sample_rate = example['samplingrate']
    winlen = int(.02 * sample_rate)                      # window length = 20ms
    winshift = int(.01 * sample_rate)                    # shift length = 10ms
    enf = enframe(samples, winlen, winshift)
    
    # Step 2: Pre-emphasis
    pre_emp = preemp(enf, p  = 0.97)
    
    # Step 3: Hamming Window
    ham = windowing(pre_emp)
    
    # Step 4: Fast Fourier Transform
    FFT = powerSpectrum(ham, nfft = 512)

    # Step 5: Mel filterbank log spectrum
    logMel = logMelSpectrum(FFT, sample_rate)
    
    # Step 6: Cosine Transofrm and Liftering
    nceps = 13
    cos_tra = cepstrum(logMel, nceps)
    l_cos_tra = tools.lifter(cos_tra)
    
    # Tidigits test for Quest 4
    l_MFCC = []
    for item in tidigits:
        tid_samples = item['samples']
        l_MFCC.append(mfcc(tid_samples))

    #------------------------------------------------ Quest 5 ----------------------------->>>
    l_MFCC_concat = np.zeros([1, nceps])
    for i in range(len(tidigits)):
        l_MFCC_concat = np.append(l_MFCC_concat, l_MFCC[i], axis = 0)
    MFCC_cor_coef = np.corrcoef(l_MFCC_concat, rowvar=0)             # rowvar = non-zero (default): each row represents a variable, with observations in the columns.
                                                                     # Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.
   
    Mspec_concat = mspec(tidigits[0]['samples']) 
    for i in range(1, len(tidigits)):
          Mspec_concat = np.append(Mspec_concat, mspec(tidigits[i]['samples']), axis=0 )
    Mspec_cor_coef = np.corrcoef(Mspec_concat, rowvar=0)
    
    #------------------------------------------------ Quest 6 ----------------------------->>>
        # check local dist function for 2 utterances from Tidigit samples
    # utter1 = mfcc(tidigits[0]['samples'])
    # utter2 = mfcc(tidigits[1]['samples'])
    # loc_dist = locdist(utter1, utter2)

        # next step
    # D = np.zeros([len(tidigits), len(tidigits) ])
    # N = len(D)               # full data: len(D)
    # M = D.shape[1]               # full data: D.shape[1]
    # for i in range(N):
    #     for j in range(M):
    #         u1 = mfcc(tidigits[i]['samples'])
    #         u2 = mfcc(tidigits[j]['samples'])
    #         loc_dist = locdist(u1, u2)
    #         D[i, j] = dtw(loc_dist)   
   
    #------------------------------------------------ Quest 7 ----------------------------->>>
    all_features = np.copy(l_MFCC_concat)
    n_components = 32
    gmm = GMM(n_components=n_components, covariance_type='full')
    feature_fit = gmm.fit(all_features)

    lift_mfcc_71 = mfcc(tidigits[16]['samples'])
    prediction_71 = gmm.predict(lift_mfcc_71)
    print("prediction 71 =", prediction_71, "\n")

    lift_mfcc_72 = mfcc(tidigits[17]['samples'])
    prediction_72 = gmm.predict(lift_mfcc_72)
    print("prediction 72 =", prediction_72, "\n")

    lift_mfcc_73 = mfcc(tidigits[38]['samples'])
    prediction_73 = gmm.predict(lift_mfcc_73)
    print("prediction 73 =", prediction_73, "\n")

    lift_mfcc_74 = mfcc(tidigits[39]['samples'])
    prediction_74 = gmm.predict(lift_mfcc_74)
    print("prediction 74 =", prediction_74, "\n")

    lift_mfcc_un = mfcc(tidigits[4]['samples'])
    prediction_un = gmm.predict(lift_mfcc_un)
    print("prediction un =", prediction_un)
    
    

    fig0 = plt.figure()
    fig0.canvas.set_window_title('num of components = ' + str(n_components))
    lab = tools.tidigit2labels(tidigits)
    plt.plot(prediction_71, 'r', label=lab[16])
    plt.legend(loc  = 'upper right')
    plt.suptitle('num of components = ' + str(n_components))
    #plt.savefig('fig13')

    fig1 = plt.figure()
    fig1.canvas.set_window_title('num of components =' + str(n_components))
    plt.plot(prediction_72, 'b', label=lab[17])
    plt.legend(loc  = 'upper right')
    plt.suptitle('num of components = ' + str(n_components))
    #plt.savefig('fig14')

    fig2 = plt.figure()
    fig2.canvas.set_window_title('num of components =' + str(n_components))
    plt.plot(prediction_73, 'g', label=lab[38])
    plt.legend(loc  = 'upper right')
    plt.suptitle('num of components = ' + str(n_components))
    #plt.savefig('fig15')

    fig3 = plt.figure()
    fig3.canvas.set_window_title('num of components =' + str(n_components))
    plt.plot(prediction_74, 'y', label=lab[39])
    plt.legend(loc  = 'upper right')
    plt.suptitle('num of components = ' + str(n_components))
    #plt.savefig('fig16')
    
    

    
    


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  VISUALIZATION  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    # # My Visualization:
    
    # -------- Quest 4
    fig1 = plt.figure(figsize=(10,10))
    fig1.canvas.set_window_title('My results')
        # Speech samples
    plt.subplot(8, 1, 1)
    plt.plot(samples)
    plt.title('Speech samples')
        # Step 1: Enframe
    plt.subplot(8, 1, 2)
    plt.pcolormesh(enf.T, cmap = 'jet')
    plt.title('Enframe')
        # Step 2: Pre-emphasis
    plt.subplot(8, 1, 3)
    plt.pcolormesh(pre_emp.T, cmap = 'jet')
    plt.title('Pre-emphasis')
        # Step 3: Hamming Window
    plt.subplot(8, 1, 4)   
    plt.pcolormesh(ham.T, cmap = 'jet')
    plt.title('Hamming Window')
        # Step 4: Fast Fourier Transform
    plt.subplot(8, 1, 5)   
    plt.pcolormesh(FFT.T, cmap = 'jet')
    plt.title('Fast Fourier Transform')
        # Step 5: Mel filterbank log spectrum
    plt.subplot(8, 1, 6)   
    plt.pcolormesh(logMel.T, cmap = 'jet')
    plt.title('Mel filterbank log spectrum')
        # Step 6: Cosine Transofrm and Liftering
    plt.subplot(8, 1, 7)   
    plt.pcolormesh(cos_tra.T, cmap = 'jet')
    plt.title('MFCC')
    plt.subplot(8, 1, 8)   
    plt.pcolormesh(l_cos_tra.T, cmap = 'jet')
    plt.title('Lifted MFCC')
        # filter banks plot
    fig2 = plt.figure()
    fig2.canvas.set_window_title('Filter Banks')
    plt.plot(fbank.T)   
    # Quest 4.6 : Tidigit utterances
    fig3 = plt.figure(figsize=(10,6))
    fig3.canvas.set_window_title('Tidigit 10 utterances: l_MFCC')
   
    plt.subplot(4, 1, 1)
    plt.title(lab[16])       
    plt.xticks([])
    plt.yticks([])
    plt.pcolormesh(l_MFCC[16].T, cmap = 'jet')
    
    plt.subplot(4, 1, 2)
    plt.title(lab[17])       
    plt.xticks([])
    plt.yticks([])
    plt.pcolormesh(l_MFCC[17].T, cmap = 'jet')
   
    plt.subplot(4, 1, 3)
    plt.title(lab[38])        
    plt.xticks([])
    plt.yticks([])
    plt.pcolormesh(l_MFCC[38].T, cmap = 'jet')
    
    plt.subplot(4, 1, 4) 
    plt.title(lab[39])      
    plt.xticks([])
    plt.yticks([])
    plt.pcolormesh(l_MFCC[39].T, cmap = 'jet')
    fig3.tight_layout(pad=2)


    # -------- Quest 5
    fig4 = plt.figure(figsize=(10,6))
    fig4.canvas.set_window_title('Feature Correlation')
    plt.subplot(2, 1, 1)
    plt.pcolormesh(MFCC_cor_coef, cmap = 'jet')

    plt.subplot(2, 1, 2)
    plt.pcolormesh(Mspec_cor_coef, cmap = 'jet')
    

    # -------- Quest 6
    # plt.pcolormesh(D, cmap = 'jet')       
    # plt.show()
    # fig5 = plt.figure(figsize=(20,30))
    # fig5.canvas.set_window_title('Comparing Utterances')
    # ax = fig5.add_subplot(111)
    # link = hac.linkage(D, method = 'complete')
    # labels = tools.tidigit2labels(tidigits)
    # dendro = hac.dendrogram(link, labels = labels)
    
    # ax.set_xticklabels(labels)
    # ax.legend(labels)
    # plt.legend(labels, loc='center', fontsize='xx-large')
  
    
   

    
    fig1.tight_layout(pad=2)
    fig2.tight_layout(pad=2)
    plt.show()                     # uncomment for visualization




    
if __name__ == "__main__":
    main()    






