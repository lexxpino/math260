import numpy as np
from numpy import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt


def tone_data():
    """ Builds the data for the phone number sounds...
        Returns:
            tones - list of the freqs. present in the phone number sounds
            nums - a dictionary mapping the num. k to its two freqs.
            pairs - a dictionary mapping the two freqs. to the nums
        Each number is represented by a pair of frequencies: a 'low' and 'high'
        For example, 4 is represented by 697 (low), 1336 (high),
        so nums[4] = (697, 1336)
        and pairs[(697, 1336)] = 4
    """
    lows = [697, 770, 852, 941]
    highs = [1209, 1336, 1477, 1633]  # (Hz)

    nums = {}
    for k in range(0, 3):
        nums[k+1] = (lows[k], highs[0])
        nums[k+4] = (lows[k], highs[1])
        nums[k+7] = (lows[k], highs[2])
    nums[0] = (lows[1], highs[3])

    pairs = {}
    for k, v in nums.items():
        pairs[(v[0], v[1])] = k

    tones = lows + highs  # combine to get total list of freqs.
    return tones, nums, pairs


def load_wav(fname):
    """ Loads a .wav file, returning the sound data.
        If stereo, converts to mono by averaging the two channels
        Returns:
            rate - the sample rate (in samples/sec)
            data - an np.array (1d) of the samples.
            length - the duration of the sound (sec)
    """
    rate, data = wavfile.read(fname)
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data[:, 0] + data[:, 1]  # stereo -> mono
    length = data.shape[0] / rate
    print(f"Loaded sound file {fname}.")
    return rate, data, length

def f(t):
    return np.sin(2 * t) - (4 * np.cos(6 * t)) + (3 * np.sin(10 * t))


def DFTwav(fname):
    """ 
    The high peak is at 1633 and the low peak is at 770, which matches
    the data from nums[0] in tone_data
    """
    rate, data, length = load_wav(fname)
    n = len(data) 
    t = np.linspace(0, length, n, endpoint=False)
    d = t[1] - t[0]
    s = [data[i] for i in range(n)]
    
    freq = fft.fftfreq(n, d)
    transform = fft.fft(s)/n
        
    tones, nums, pairs = tone_data()
   
    dualplot(freq, transform, "DFT (real/imag parts)")
    plt.show()
    
    
def dualplot(freq, sf, name):
    """ plot of real and imaginary parts """
    plt.figure(figsize=(6.5, 2.5))
    plt.suptitle(name)
    plt.subplot(1, 2, 1)
    plt.plot(freq, np.real(sf), '.k')
    plt.ylabel('Re(F)')
    plt.subplot(1, 2, 2)
    plt.plot(freq, np.imag(sf), '.k')
    plt.ylabel('Im(F)')
    plt.subplots_adjust(wspace=0.5)

    
def id_dial(fname):
    tone_length = 0.7  # signal broken into 0.7 sec chunks
    rate, data, sound_length = load_wav(fname)
    tones, nums, pairs = tone_data()
    
    kek = ""
    for i in range(9):
        kek += str(nums[i])
    print(kek)
    
    dialed = [-1]*7
    
    start = 0
    end = start + int(len(data)/7)
    
    n = int(len(data)/7) # number of samples per chunk
    t = np.linspace(0, tone_length, n, endpoint=False)
    d = t[1] - t[0]
    for k in range(7):
        start = int(k * (len(data)/7))
        end = start + int(len(data)/7)
        s = [data[j] for j in range(start, end)]
    
        freq = fft.fftfreq(n, d)
        transform = fft.fft(s)/n 
        
        for digit in range(10):
            low = nums[digit][0]
            high = nums[digit][1]
            
            # low or high = freq[k] = k/L
            low_k = int(low * tone_length)
            high_k = int(high * tone_length)
            
            reals = np.real(transform)
            real_low = max(reals[low_k - 1], reals[low_k], reals[low_k + 1])

            real_high = max(reals[high_k - 1], reals[high_k], reals[high_k + 1])

            
            low_peak = True
            for i in range(low_k - 50, min(n, low_k + 50)):
                if reals[i] > real_low:
                    low_peak = False
                    
            high_peak = True
            for i in range(high_k - 50, min(n, high_k + 50)):
                if reals[i] > real_high:
                    high_peak = False
            
            if low_peak ==True and high_peak==True:
                dialed[k] = digit
                break
    
    print(dialed)      
    

def identify_dig(fname):
    rate, data, length = load_wav(fname)
    n = len(data)
    s = [data[i] for i in range(n)]

    # take fft of "data"...
    transform = fft.fft(s) / n
    real_transform = abs(transform)

    tones, nums, pairs = tone_data()

    # analyze transform to find digit
    for digit in range(10):
        low = nums[digit][0]
        high = nums[digit][1]

        #freq[k] = k / L --> l * freq[k] = k
        low_k = int(low * length)
        high_k = int(high * length)

        low_peak = True
        high_peak = True
        
        #check if this is a peak in the graph
        for i in range(low_k - 20, low_k + 20):
            if real_transform[i] > real_transform[low_k] and real_transform[i] > real_transform[low_k + 1]:
                low_peak = False
        
        #check if this is a peak in the graph
        for i in range(high_k - 20, high_k + 20):
            if real_transform[i] > real_transform[high_k] and real_transform[i] > real_transform[high_k + 1]:
                high_peak = False

        if low_peak and high_peak:
            return digit

    #No digit found
    return None
    

if __name__ == "__main__":
    DFTwav("hw9/0.wav")
    DFTwav("hw9/5.wav")

    print(identify_dig('hw9/0.wav'))
    print(identify_dig('hw9/5.wav'))

    id_dial('hw9/dial.wav')
    id_dial('hw9/dial2.wav')
    id_dial('hw9/noisy_dial.wav')
