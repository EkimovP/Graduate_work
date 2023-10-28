from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSizePolicy
import sys
from window import *
from calculating_funcitons import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from multiprocessing import Process, Manager

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, cols=2, rows=2, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.cols = cols
        self.rows = rows
        self.axes = {}
        super(MplCanvas, self).__init__(self.fig) 
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
    
    def add_plot(self, row, col, name):
        self.axes[name] = self.fig.add_subplot(self.rows,self.cols,col + self.cols * (row-1))

class WidgetPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = MplCanvas(kwargs["parent"], width=5, height=4)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

class SpectrumAnalysWin(QMainWindow):
    def __init__(self):
        super(SpectrumAnalysWin,self).__init__() #вызов __init__ родителя
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()

        self.fd = 300e3
        self.sins = [Sinusoida(1,40e3,0,self.fd)]
        self.t_max = 10e-3
        self.SNR = 10
        self.spectrum = []
        self.win_step = 8
        self.win_size = 256
        self.intervals = [(0.4, 0.6)]
        self.mode = "FFT" 
        self.connect()
        self.update_data(False)
        self.generate_sins()
        
    def initUI(self):
        """
        Размещение виджетов на окне
        """
        self.plot = MplCanvas(self, rows=3, cols=2, width=8, height=6, dpi=100)
        self.toolbar = NavigationToolbar2QT(self.plot, self)
        self.plot.add_plot(1, 1, "signal")
        self.plot.add_plot(2, 1, "spectrum")
        self.plot.add_plot(1, 2, "spectrogram")
        self.plot.add_plot(2, 2, "K")
        self.plot.add_plot(3, 1, "rho")
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.plot)

        # Create a placeholder widget to hold our toolbar and canvas.
        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(self.layout)
        self.widget.show()
        self.plot.axes["signal"].plot([0,1,2,3,4], [10,1,20,3,40])
        self.plot.axes["signal"].set_title("График сигнала")
        self.plot.axes["signal"].set_xlabel("t, c")
        self.plot.axes["signal"].set_ylabel("A")

        self.ui.FFTradioButton.setChecked(True)

        #тулбар для графиков
        
    def connect(self):
        self.ui.generateSignalButton.clicked.connect(self.generate_sins)
        self.ui.generateSpectrogramButton.clicked.connect(
                                                          lambda:self.calculate_spectrum() 
                                                          or self.calculate_spectrogram()
                                                          or self.calculate_K()
                                                          )
        self.ui.generateRhoButton.clicked.connect(self.calculate_rho)

    def update_data(self, save = True):
        if save:
            self.fd = float(self.ui.fdEdit.text())
            self.t_max = float(self.ui.tMaxEdit.text())
            self.SNR = float(self.ui.SnrEdit.text())
            self.win_step = float(self.ui.winStepEdit.text())
            self.win_size = float(self.ui.winSizeEdit.text())
            
        else:
            self.ui.fdEdit.setText(str(self.fd))
            self.ui.tMaxEdit.setText(str(self.t_max))
            self.ui.SnrEdit.setText(str(self.SNR))
            self.ui.winSizeEdit.setText(str(self.win_size))
            self.ui.winStepEdit.setText(str(self.win_step))

        self.update_sins(save)
        self.update_mode()
        
    def update_sins(self, save = True):
        if save:
            params = [float(param) for param in self.ui.sinParamEdit.text().split()]
            n_params = len(params)
            self.sins = [
                    Sinusoida(params[i%3], params[i%3 + 1], params[i%3 + 2], self.fd)
                    for i in range(n_params//3)
                            ]
        else:
            params = [sin.get_params() for sin in self.sins]
            params_str = " ".join(params)
            self.ui.sinParamEdit.setText(params_str)

    def update_mode(self):
        for radioButton in [self.ui.AKFradioButton, self.ui.ARradioButton, self.ui.FFTradioButton, self.ui.MMDradioButton]:
            if radioButton.isChecked():
                self.mode = radioButton.text()

    def generate_sins(self):
        self.update_data()
        self.signal, self.signal_dots = generate_signal(self.t_max,
                                                        self.fd,
                                                        self.sins,
                                                        self.intervals)
        self.signal = add_noise(self.signal, self.SNR)
        self.plot.axes["signal"].cla()
        self.plot.axes["signal"].plot(self.signal_dots, self.signal)
        self.plot.draw()

    def calculate_spectrum(self):
        self.update_data()
        self.spectrum = calculate_spectrum(self.signal, self.fd, self.SNR, self.mode)
        self.spectrum_dots = generate_spectrum_dots(len(self.signal), self.fd)
        n = self.spectrum_dots.size
        self.plot.axes["spectrum"].cla()
        self.plot.axes["spectrum"].plot(self.spectrum_dots[:n//2], abs(self.spectrum[:n//2]))
        self.plot.draw()
        
    def calculate_spectrogram(self):
        self.update_data()
        self.spectrogram_x, self.spectrogram_y, self.spectrogram = calculate_spectrogram(self.signal, 
                                                                                         self.fd, 
                                                                                         self.win_size, 
                                                                                         self.win_step, 
                                                                                         self.SNR, 
                                                                                         self.mode)
        self.plot.axes["spectrogram"].cla()
        self.plot.axes["spectrogram"].contourf(self.spectrogram_x, self.spectrogram_y, abs(self.spectrogram), levels=100)
        self.plot.draw()    

    def calculate_K(self):
        self.update_data()
        self.K = calculate_K(self.spectrogram, self.sins[0].w, self.fd / 6, self.fd)
        self.plot.axes["K"].cla()
        self.plot.axes["K"].plot(self.spectrogram_x, abs(self.K))
        self.plot.draw()

    def calculate_rho(self):
        self.update_data()
        clean_signal, _ = generate_signal(self.t_max, self.fd, self.sins, self.intervals)
        self.rho_x, self.rho = calculate_rho_array1(clean_signal, self.fd, self.win_size, self.win_step, self.SNR,
                                       self.mode, self.sins[0].w, self.intervals)
        self.plot.axes["rho"].cla()
        self.plot.axes["rho"].plot(self.rho_x, self.rho)
        self.plot.draw()

def calculate_rho_array_proccess(clean_signal:np.ndarray, fd, win_size, win_step, SNR, mode, f, intervals, N, N_start, N_end, rho):
    for i in np.arange(N_start, N_end):
        threshold = i / N
        calculate_rho(clean_signal, fd, win_size, win_step, SNR, mode, f, threshold, intervals, rho, i)      
        
def calculate_rho_array(clean_signal:np.ndarray, fd, win_size, win_step, SNR, mode, f, intervals):
    N = 200
    processes_n = 6
    processes = []
    manager = Manager()
    rho = manager.dict()
    for i in np.arange(processes_n):
        N_start = int(i * N / processes_n)
        N_end = int((i + 1) * N / processes_n)
        p = Process(target=calculate_rho_array_proccess, args=(clean_signal, fd, win_size, win_step, SNR, mode, f, intervals, N, N_start, N_end, rho))
        processes.append(p)
        p.start()
    for i, p in enumerate(processes):
        p.join()
    x = np.array([i / N for i in np.arange(N)])
    rho = np.array([value for key, value in rho.items()])
    rho.sort()
    return x, rho

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SpectrumAnalysWin()
    win.show()
    sys.exit(app.exec())