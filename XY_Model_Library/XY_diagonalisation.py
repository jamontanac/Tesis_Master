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
    def Sample_State(cls,Ground:bool =False,mu:np.float64 = 0.0)-> np.ndarray:

        if Ground:
            x=np.arange(0,(cls.N_size-1)/2+ 1)
            m_cos=[-0.5 for i in x]
            m_sin=[-0.5 for i in x]
            x=np.arange(-(cls.N_size-1)/2,(cls.N_size-1)/2+1)
            M_minous=[((m_cos[np.abs(int(i))]-m_sin[np.abs(int(i))])*0.5*np.exp(1.j*np.sign((2.0*np.pi/cls.N_size) * i)*cls.Phi(np.abs((2.0*np.pi/cls.N_size) * i)))) for i in x]
            M_plus = [((m_cos[np.abs(int(i))]+m_sin[np.abs(int(i))])*0.5*np.exp(1.j*np.sign((2.0*np.pi/cls.N_size) * i)*cls.Phi(np.abs((2.0*np.pi/cls.N_size) * i)))) for i in x]
            Mminousband=np.array(M_minous)
            Mplusband=np.array(M_plus)
        else:
            x=np.arange(0,(cls.N_size-1)/2+ 1)
            m_cos=[-0.5 if np.random.random()>cls.Fermi_dirac(mu=mu,n=i) else 0.5 for i in x]
            m_sin=[-0.5 if np.random.random()>cls.Fermi_dirac(mu=mu,n=i) else 0.5 for i in x]
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
        Fourier_plus=pyfftw.empty_aligned(cls..N_size, dtype='complex128')
        Fourier_plus[:]=np.fft.ifftshift(Mplus)
        Fourier_minous=pyfftw.empty_aligned(cls..N_size, dtype='complex128')
        Fourier_minous[:]=np.fft.ifftshift(Mminous)

    #
    #     x=np.arange(-(N_size-1)/2,(N_size-1)/2+ 1)
    #     Fourier_plus=fft(ifftshift(Mplusband))
    #     Fourier_minous=fft(ifftshift(Mminousband))
    #     return Fourier_plus/N_size,Fourier_minous/N_size


a= Sampling_Random_State()
uno,dos = a.Sample_State()
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
