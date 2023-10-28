import numpy as np
import kaczmarz
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import welch

class Sinusoida:
    def __init__(self, A, w, phi, fd = 300e3):
        self.A = A
        self.w = w
        self.phi = phi
        self.fd = fd

    def get_value(self, t):
        return self.A * np.sin(2. * np.pi * self.w * t + self.phi + self.FM2phi(t, self.fd))    
    
    def get_params(self):
        return f"{self.A} {self.w} {self.phi}"
    
    def FM2phi(self, t, fd):
        #t приводим от 0 до 1
        t -= int(t);
        bits = 9600;
        t *= bits;
        if (int(t) % 2):
            return np.pi
        return 0;    


def generate_signal(t_max, fd, sins, intervals: list = None):
    """
    функция возвращает сигнал и его временные отсчеты
    """
    if intervals == None:
        intervals = [(0,t_max)]
    signal_dots = np.array([t for t in np.arange(0, t_max, 1/fd)])
    signal = np.zeros(signal_dots.size)
    for index_start, index_end in intervals:
        t_start = t_max * index_start
        t_end = t_max * index_end
        signal += np.array([sum([sin.get_value(t) for sin in sins]) 
                            if t >= t_start and t <= t_end else 0
                        for t in signal_dots])
    return signal, signal_dots

def add_noise(signal:np.array, SNR):
    noiseAmpl = np.sqrt(1 / pow(10, SNR / 10.))
    noise = np.random.normal(0,noiseAmpl,signal.size)
    return signal + noise

def generate_spectrum_dots(signal_size, fd):
    spectrum_dots = np.array([i * fd / signal_size for i in np.arange(signal_size)])
    return spectrum_dots

def calculate_spectrum(signal:np.ndarray, fd, SNR, mode="FFT"):
    if mode=="FFT":
        data = np.fft.fft(signal)
    elif mode == "AR":
        data = calculate_AR_spectrum(signal, 10, SNR)
    elif mode == "MMD":
        data = SpectrumMMD(signal, fd, 8, SNR)
    elif mode == "AKF":
        AKF = np.correlate(signal,signal,mode="full")
        AKF = AKF[len(AKF)//2+1:]
        n = len(AKF)
        AKF = np.concatenate((AKF[:n//8], [0]*(n - n//8)))
        data = np.fft.fft(AKF)
    return data

def calculate_spectrogram(signal: np.ndarray, fd, win_size, win_step, SNR, mode="FFT"):
    data = []
    x = []
    y = generate_spectrum_dots(win_size, fd)
    
    for ind_t in np.arange(0, len(signal) - win_size, win_step):
        subsignal = signal[int(ind_t) : int(ind_t + win_size)]
        data.append(calculate_spectrum(subsignal, fd, SNR, mode))
        x.append(ind_t / fd)
    x = np.array(x)
    data = np.array(data)
    return (x,
            y[:int(win_size//2)],
            data.transpose()[:int(win_size//2),:]) #обрезали половину спектра

def caluculate_Rxx(signal: np.ndarray, range, SNR):
    rxx = np.correlate(signal, signal, "full")
    rxx = rxx[len(rxx)//2+1:]
    Rxx = np.array([[rxx[abs(i - j)] for j in np.arange(range + 1)]for i in np.arange(range + 1)])
    noiseAmpl = np.sqrt(1 / pow(10, SNR / 10.))
    #шум на диагонали
    for i in np.arange(range + 1):
        Rxx[i][i] += np.random.normal(0,noiseAmpl * 0.1)
    return Rxx

def caluculate_AR_coeffs(Rxx: np.array):
    range = len(Rxx)
    y = [0] * range
    y[0]= Rxx[0][0] * 0.25
    return kaczmarz.Cyclic.solve(Rxx,y)

def calculate_AR_f(a: np.ndarray, f, fd):
    n = a.size
    e = []
    I = 0.0+1.j
    e = np.array([np.exp(I * 2 * np.pi * f * n * i / fd) for i in np.arange(n)])
    eH = e.conj().T
    aH = a.conj().T 
    e.shape = (n, 1)
    a.shape = (n, 1)
    aH.shape = (1, n)
    eH.shape = (1, n)
    return 1 / (eH @ a @ aH @ e)

def calculate_AR_spectrum(signal: np.ndarray, range):
    # Обучение AR-модели
    model = AutoReg(signal, lags = range)
    model_fit = model.fit(cov_kwds={"maxlags":range})
    # Вычисление спектра
    spectrum = np.real(np.fft.fft(model_fit.params, n=signal.size)) 
    #TODO
    return np.exp(spectrum)

def MMD(Rxx, f, fd):
    n = len(Rxx)
    I = 0.0+1.j
    e = np.array([np.exp(I * 2 * np.pi * f * n * i / fd) for i in np.arange(n)])
    e[0] = 1
    eH = e.conj().T
    e.shape = (n, 1)
    eH.shape = (1, n)
    inverse_Rxx = np.linalg.inv(Rxx)
    return 1 / (eH @ inverse_Rxx @ e)

def SpectrumMMD(signal, fd):
    Pxx = welch(signal, fd, nperseg=fd/2, noverlap=8)
    #TODO
    return Pxx

def calculate_K(spectrogram: np.ndarray, f, delta, fd):
    K = []
    f_start = f - delta
    if f_start < 0:
        f_start = 0
    f_end = f + delta
    index_start = int(f_start * 2 * len(spectrogram) / fd)  # *2 - ибо в спектрограмме только половина fd
    index_end = int(f_end * 2 * len(spectrogram) / fd)  # *2 - ибо в спектрограмме только половина fd
    for i in np.arange(len(spectrogram[0])):
        band = spectrogram[:,i]
        band = np.array(np.abs(band) * np.abs(band) ,dtype=np.float64)
        all_energy = band.reshape((1,-1)) @ band.reshape((-1,1))
        subband = band[index_start:index_end+1]
        signal_energy = subband.reshape((1,-1)) @ subband.reshape((-1,1))
        value = (np.abs(signal_energy) / np.abs(all_energy)).ravel()[0]
        #TODO
        if all_energy < 1:
            K.append(value**2)
        else:
            K.append(value)
    return np.array(K).ravel()
        
def check_threshold(K: np.ndarray, threshold, intervals):
    n_start = int(K.size * intervals[0][0] * 0.90)
    n_end = int(K.size * intervals[0][1] * 1.1)
    result = K >= threshold
    #за пределами сигнала обнаружена хотя бы одна точка, будто из сигнала - неверно обнаружен
    if np.any(result[:n_start]) or np.any(result[n_end+1:]):
        return 0
    #19% точек достаточно, чтобы считать обнаруженным
    return sum(result[n_start:n_end+1]) > ((n_end - n_start) * 0.19)

#точка на графике вероятности обнаружения. то есть, rho - вероятность верного обнаружения сигнала
def calculate_rho(clean_signal, fd, win_size, win_step, SNR, mode, f, threshold, intervals, results, index):
    count = 0
    #TODO
    N = 10
    for i in np.arange(N):
        signal = add_noise(clean_signal, SNR)
        _, _, spectrogram = calculate_spectrogram(signal,fd, win_size, win_step, SNR, mode)
        K = calculate_K(spectrogram, f, fd/6, fd)
        count += check_threshold(K, threshold, intervals)
    results[index] = count / N
    
def calculate_rho_array1(clean_signal:np.ndarray, fd, win_size, win_step, SNR, mode, f, intervals):
    #TODO
    N = 100
    rho = [0] * N
    for i in np.arange(N-1, -1, -1):
        threshold = i / N
        calculate_rho(clean_signal, fd, win_size, win_step, SNR, mode, f, threshold, intervals, rho, i)
    x = np.array([i / N for i in np.arange(N)])
    rho = np.array(rho)
    return x, rho