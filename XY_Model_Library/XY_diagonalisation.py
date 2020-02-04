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
    def Fermi_dirac(cls,n:np.int64,beta:np.float64,mu:np.float64 =0) -> np.float64:
        # beta is the inverse thermic energy associated in the system (beta)
        # mu corresponds to the chemical potential
        # n is the position of the particle
        # f=np.exp(T*(Omega(Gamma,Lambda,2.0*(np.pi/N)*n)-mu)) +1
        # N corresponds to the size of the system

        f=np.exp(beta*(cls.Omega(((2.*np.pi)/np.float64(cls.N_size)) * n)-mu)) +1
        return 1/f
    @classmethod
    def Sample_number_sin_cos(cls,Ground:bool = False, mu : np.float64 =0.0)-> list:
        x=np.arange(0,(cls.N_size-1)/2+ 1)
        beta = np.min(cls.Omega(np.linspace(-np.pi,np.pi,int(1000))))
        if Ground:
            m_cos=[-0.5 for i in x]
            m_sin=[-0.5 for i in x]
        else:
            m_cos=[-0.5 if np.random.random()>cls.Fermi_dirac(mu=mu,n=i,beta=beta) else 0.5 for i in x]
            m_sin=[-0.5 if np.random.random()>cls.Fermi_dirac(mu=mu,n=i,beta=beta) else 0.5 for i in x]
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
    def Hankel_matrix(cls,Fourier_M:np.ndarray,L:np.int64)-> np.ndarray:
        to_use=Fourier_M[:2*L-1]
        First_column=to_use[:L]
        Last_row=np.roll(to_use,-L+1)[:L]
        return hankel(First_column,Last_row)

    @classmethod
    def Covariance_matrix(cls,L:np.int64,mu:np.float64=0.0,Ground:bool=False)-> np.ndarray:
        Fourier_minous,Fourier_plus=cls.Get_Bands_Matrix(mu=mu,Ground=Ground)
        return (cls.Toeplitz_matrix(Fourier_plus,L)+cls.Hankel_matrix(Fourier_minous,L))
    @classmethod
    def Covariance_matrix_from_sub_sample(cls,Fourier_plus:np.ndarray,Fourier_minous:np.ndarray,L:np.int64)-> np.ndarray:
        return (cls.Toeplitz_matrix(Fourier_plus,L)+cls.Hankel_matrix(Fourier_minous,L))

    @classmethod
    def get_band_of_matrix(cls,Matrix:np.ndarray,num_band:np.int64)-> np.ndarray:
        L,C=Matrix.shape
        if L!=C:
            raise ValueError("Only squared matrix can be computed")
        if num_band > 0:
            return np.array([[Matrix[i,j] for i in range(num_band,L) if i-j == num_band] for j in range(L-num_band)]).reshape(L-num_band)
        elif num_band <0:
            return np.array([[Matrix[i,j] for i in range(L) if i-j == num_band] for j in range(-num_band,L)]).reshape(L+num_band)
        else:
            return np.diagonal(Matrix)


def Fermi_dirac(n:np.int64,beta:np.float64,Size:np.int64,mu:np.float64 =0.0) -> np.float64:
    # beta is the inverse thermic energy associated in the system (beta)
    # mu corresponds to the chemical potential
    # n is the position of the particle
    # f=np.exp(T*(Omega(Gamma,Lambda,2.0*(np.pi/N)*n)-mu)) +1
    # N corresponds to the size of the system
    instance = Sampling_Random_State()
    f=np.exp(beta*(instance.Omega(((2.*np.pi)/np.float64(Size)) * n)-mu)) +1
    return 1/f

State = Sampling_Random_State()
F_minous,F_plus=State.Get_Bands_Matrix()
beta = np.min(State.Omega(np.linspace(-np.pi,np.pi,int(1000))))
L=40
New_cov_matrix=State.Covariance_matrix_from_sub_sample(F_plus,F_minous,L)
S=np.linalg.svd(New_cov_matrix,compute_uv=False)
n=np.arange(-(L-1)/2,(L-1)/2 +1)
array_to_plot=sorted(-S+0.5,reverse=True)
plt.plot(array_to_plot,label="Singular values")
plt.plot(np.array(sorted(Fermi_dirac(n=n,Size=L,beta=beta),reverse=True)),label="Fermi distribution")
plt.legend()
plt.title("lenght of {}".format(L))
plt.show()