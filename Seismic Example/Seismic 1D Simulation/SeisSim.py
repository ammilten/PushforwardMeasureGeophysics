
# -----------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate 
import math

def reflection_coeff(rho1, V1, rho2, V2):
    Z1 = rho1*V1
    Z2 = rho2*V2
    return (Z1-Z2)/(Z1+Z2)

def ricker(t, fM):
    A = (1-2*np.square(math.pi*fM*t))
    B = np.exp(-1*np.square(math.pi*fM*t))
    return A*B


class LayerSeismic:
    def __init__(self, rhob=2000, Vb=3000, rhol=1500, Vl=1500, h=200, dh=100, fM=25, T=1.0, dt=0.002, outfile=None):
        self.rhob = rhob
        self.Vb = Vb
        self.rhol = rhol
        self.Vl = Vl
        self.h = h
        self.dh = dh
        self.fM = fM
        self.tmax = T
        self.dt = dt
        self.outfile = outfile

    def prepare_reflection_series(self):
        self.tR = np.arange(0,self.tmax,self.dt)
        t1 = 2 * self.h/self.Vb
        t2 = t1 + 2 * self.dh/self.Vl

        t1_ind = np.argmin(np.abs(t1-self.tR))
        t2_ind = np.argmin(np.abs(t2-self.tR))
        
        R = np.zeros(self.tR.shape[0])
        R[t1_ind] = reflection_coeff(self.rhob, self.Vb, self.rhol, self.Vl)
        R[t2_ind] = reflection_coeff(self.rhol, self.Vl, self.rhob, self.Vb)
        self.R = R

        return R

    def prepare_wavelet(self):
        tmax = self.tR[self.tR.shape[0]-1]/4
        self.tw = np.arange(-tmax, tmax, self.dt)
        self.wavelet = ricker(self.tw, self.fM)
        return self.wavelet

    def run_convolution(self):
        waveform = np.convolve(self.R, self.wavelet)
        t = np.arange(waveform.shape[0]) * self.dt - self.wavelet.shape[0]/2* self.dt
        t0_ind = np.argmin(np.abs(t))
        tf_ind = waveform.shape[0]-t0_ind
        self.seismic = waveform[t0_ind:tf_ind]
        self.ts = t[t0_ind:tf_ind]

        if self.outfile is not None:
            outdata = np.concatenate((self.ts[:,np.newaxis],self.seismic[:,np.newaxis]),axis=1)
            np.savetxt(self.outfile, outdata, delimiter=',',header="Two-Way Travel Time (s), Seismic Amplitude")
        
        return self.seismic

    def show_data(self, figsize=(20,8)):
        plt.subplots(nrows=1, ncols=6, figsize=figsize)
 
        Z = np.array([0, self.h, self.h, self.h+self.dh, self.h+self.dh, self.H])
        rho = np.array([self.rhob, self.rhob, self.rhol, self.rhol, self.rhob, self.rhob])
        V = np.array([self.Vb, self.Vb, self.Vl, self.Vl, self.Vb, self.Vb])
        I = rho*V
     
        plt.subplot(161)
        plt.plot(rho, Z)
        plt.gca().invert_yaxis()
        plt.ylabel('Depth (m)')
        plt.xlabel('Density (kg/m^3)')
        plt.title('Density Profile')

        plt.subplot(162)
        plt.plot(V, Z)
        plt.gca().invert_yaxis()
        plt.ylabel('Depth (m)')
        plt.xlabel('Velocity (m/s)')
        plt.title('P-Wave Velocity Profile')

        plt.subplot(163)
        plt.plot(I, Z)
        plt.gca().invert_yaxis()
        plt.ylabel('Depth (m)')
        plt.xlabel('Impedance (kg/m^2/s)')
        plt.title('Impedance Profile')

        plt.subplot(164)
        plt.plot(self.R, self.tR)
        plt.gca().invert_yaxis()
        plt.xlabel('Reflection Coefficient')
        plt.ylabel('Two-Way Travel Time (s)')
        plt.title('Reflection Series')

        plt.subplot(165)
        plt.plot(self.wavelet, self.tw)
        plt.gca().invert_yaxis()
        plt.xlabel('Amplitude')
        plt.ylabel('Time (s)')
        plt.title('Source Wavelet')

        plt.subplot(166)
        plt.plot(self.seismic, self.ts)
        plt.gca().invert_yaxis()
        plt.xlabel('Seismic Amplitude')
        plt.ylabel('Two-Way Travel Time (s)')
        plt.title('Seismogram')

        plt.tight_layout()
        plt.show()

    def run_full(self, showPlot=False, verbose=False):
        if verbose:
            print('Calculating reflection series...')
        self.prepare_reflection_series()

        if verbose:
            print('Preparing wavelet...')
        self.prepare_wavelet()
        
        if verbose:
            print('Running convolution model...')
        self.run_convolution()

        if showPlot:
            self.show_data()
        
        return 
    


# ------------------------------------------
if __name__ == "__main__":

    seis = LayerSeismic()
    seis.run_full(verbose=False, showPlot=True)






