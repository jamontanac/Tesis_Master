import numpy as np
import matplotlib.pylab as plt
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pyfftw
from scipy.linalg import circulant,toeplitz, hankel, expm
from matplotlib import colors
import pickle
import sys
#from scipy.fftpack import fft, ifft,ifftshift,fftshift
#from IPython.display import display, HTML


class Sampling_Random_State:

    #### --------- Definition of variables ------------------------
    N_size=500001
    Gamma=0.5
    Lambda=0.5
    num_data = 2000
    #### ------------------------------------------------------------


    @classmethod
    def Alpha(cls,theta:np.float64)-> np.float64:
        return cls.Lambda+np.cos(theta)
    @classmethod
    def Beta(cls,theta:np.float64)-> np.float64:
        return cls.Gamma*np.sin(theta)
    @classmethod
    def Omega(cls,theta:np.float64)-> np.float64:
        return np.sqrt(cls.Alpha(theta)**2 + cls.Beta(theta)**2 )
    @classmethod
    def Phi(cls,theta:np.float64)-> np.float64:
        return np.arctan2(cls.Beta(theta),cls.Alpha(theta))



    @classmethod
    def Fermi_dirac(cls,n:np.int64,mu:np.float64 =0) -> np.float64:
        # beta is the inverse thermic energy associated in the system (beta)
        # mu corresponds to the chemical potential
        # n is the position of the particle
        # f=np.exp(T*(Omega(Gamma,Lambda,2.0*(np.pi/N)*n)-mu)) +1
        # N corresponds to the size of the system
        beta = np.min(cls.Omega(np.linspace(-np.pi,np.pi,int(1000))))
        f=np.exp(beta*(cls.Omega(((2.*np.pi)/np.float64(cls.N_size)) * n)-mu)) +1
        return 1/f
    @classmethod
    def Sample_number_sin_cos(cls,Ground:bool = False, mu : np.float64 =0.0)-> list:
        x=np.arange(0,(cls.N_size-1)/2+ 1)
        if Ground:
            m_cos=[-0.5 for i in x]
            m_sin=[-0.5 for i in x]
        else:
            m_cos=[-0.5 if np.random.random()>cls.Fermi_dirac(mu=mu,n=i) else 0.5 for i in x]
            m_sin=[-0.5 if np.random.random()>cls.Fermi_dirac(mu=mu,n=i) else 0.5 for i in x]
        return m_sin,m_cos
    @classmethod
    def Sample_State(cls,Ground:bool =False,mu:np.float64 = 0.0)-> np.ndarray:
        m_sin,m_cos = cls.Sample_number_sin_cos(Ground=Ground,mu=mu)
        x=np.arange(-(cls.N_size-1)/2,(cls.N_size-1)/2+1)
        M_minous=[((m_cos[np.abs(int(i))]-m_sin[np.abs(int(i))])*0.5*np.exp(1.j*np.sign((2.0*np.pi/cls.N_size) * i)*cls.Phi(np.abs((2.0*np.pi/cls.N_size) * i)))) for i in x]
        M_plus = [((m_cos[np.abs(int(i))]+m_sin[np.abs(int(i))])*0.5*np.exp(1.j*np.sign((2.0*np.pi/cls.N_size) * i)*cls.Phi(np.abs((2.0*np.pi/cls.N_size) * i)))) for i in x]
        Mminousband=np.array(M_minous)
        Mplusband=np.array(M_plus)
        return Mminousband,Mplusband

    @classmethod
    def Get_Bands_Matrix(cls,Ground:bool =False,mu:np.float64 = 0.0)-> np.ndarray:
        Mminous, Mplus = cls.Sample_State(Ground=Ground,mu=mu)
        x=np.arange(-(cls.N_size-1)/2,(cls.N_size-1)/2+ 1)
        M_plus=pyfftw.empty_aligned(cls.N_size, dtype='complex128')
        M_plus[:]=np.fft.ifftshift(Mplus)
        M_minous=pyfftw.empty_aligned(cls.N_size, dtype='complex128')
        M_minous[:]=np.fft.ifftshift(Mminous)
        Fourier_minous=pyfftw.interfaces.numpy_fft.fft(M_minous)
        Fourier_plus=pyfftw.interfaces.numpy_fft.fft(M_plus)
        return Fourier_minous/cls.N_size, Fourier_plus/cls.N_size

    @classmethod
    def Toeplitz_matrix(cls,Fourier_P:np.ndarray,L:np.int64)-> np.ndarray:
        First_column = Fourier_P[:L]
        First_row = np.roll(Fourier_P,-1)[::-1][:L]
        return toeplitz(First_column,First_row)

    @classmethod
    def Hankel_matrix(cls,Fourier_minous:np.ndarray,L:np.int64)-> np.ndarray:
        to_use=Fourier_minous[:2*L-1]
        First_column=to_use[:L]
        Last_row=np.roll(to_use,-L+1)[:L]
        return hankel(First_column,Last_row)

    #
    #     x=np.arange(-(N_size-1)/2,(N_size-1)/2+ 1)
    #     Fourier_plus=fft(ifftshift(Mplusband))
    #     Fourier_minous=fft(ifftshift(Mminousband))
    #     return Fourier_plus/N_size,Fourier_minous/N_size


a= Sampling_Random_State()
uno,dos = a.Get_Bands_Matrix()
plt.plot(uno.real)
plt.show()
plt.plot(dos.real)
plt.show()

#plt.show()
# import time
#
# init = time.time()
# l1=np.min(a.Omega(np.linspace(-np.pi,np.pi,int(500000))))
# print("time = {}".format(time.time()-init))
#
# init = time.time()
# l2=np.min(a.Omega(np.linspace(-np.pi,np.pi,int(1000))))
# print("time = {}".format(time.time()-init))
# print(l1-l2,l1,l2)
