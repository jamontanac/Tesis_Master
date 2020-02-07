import numpy as np
from scipy.linalg import circulant,toeplitz, hankel, expm
import warnings
import pickle
try:
    import pyfftw
except ImportError:
    warnings.warn("I couldn't find the package pyfftw, please check the location of it and re run it. Our suggestion is to install it to use the functions that have been optimized")



class Sampling_Random_State:

    #### --------- Definition of variables ------------------------
    N_size=500001
    Gamma=0.5
    Lambda=0.5
    num_data = 200
    mu = 0.0
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
    def Fermi_dirac(cls,n:np.int64,beta:np.float64) -> np.float64:
        # beta is the inverse thermic energy associated in the system (beta)
        # mu corresponds to the chemical potential
        # n is the position of the particle
        # f=np.exp(T*(Omega(Gamma,Lambda,2.0*(np.pi/N)*n)-mu)) +1
        # N corresponds to the size of the system

        f=np.exp(beta*(cls.Omega(((2.*np.pi)/np.float64(cls.N_size)) * n)-cls.mu)) +1
        return 1/f
    @classmethod
    def Sample_number_sin_cos(cls,Ground:bool = False)-> list:
        x=np.arange(0,(cls.N_size-1)/2+ 1)
        beta = np.min(cls.Omega(np.linspace(-np.pi,np.pi,int(1000))))
        if Ground:
            m_cos=[-0.5 for i in x]
            m_sin=[-0.5 for i in x]
        else:
            m_cos=[-0.5 if np.random.random()>cls.Fermi_dirac(n=i,beta=beta) else 0.5 for i in x]
            m_sin=[-0.5 if np.random.random()>cls.Fermi_dirac(n=i,beta=beta) else 0.5 for i in x]
        return m_sin,m_cos
    @classmethod
    def Sample_State(cls,Ground:bool =False)-> np.ndarray:
        m_sin,m_cos = cls.Sample_number_sin_cos(Ground=Ground)
        x=np.arange(-(cls.N_size-1)/2,(cls.N_size-1)/2+1)
        M_minous=[((m_cos[np.abs(int(i))]-m_sin[np.abs(int(i))])*0.5*np.exp(1.j*np.sign((2.0*np.pi/cls.N_size) * i)*cls.Phi(np.abs((2.0*np.pi/cls.N_size) * i)))) for i in x]
        M_plus = [((m_cos[np.abs(int(i))]+m_sin[np.abs(int(i))])*0.5*np.exp(1.j*np.sign((2.0*np.pi/cls.N_size) * i)*cls.Phi(np.abs((2.0*np.pi/cls.N_size) * i)))) for i in x]
        Mminousband=np.array(M_minous)
        Mplusband=np.array(M_plus)
        return Mminousband,Mplusband

    @classmethod
    def Get_Bands_Matrix_local(cls,Ground:bool =False)-> np.ndarray:
        Mminous, Mplus = cls.Sample_State(Ground=Ground)
        x=np.arange(-(cls.N_size-1)/2,(cls.N_size-1)/2+ 1)
        M_plus=pyfftw.empty_aligned(cls.N_size, dtype='complex128')
        M_plus[:]=np.fft.ifftshift(Mplus)
        M_minous=pyfftw.empty_aligned(cls.N_size, dtype='complex128')
        M_minous[:]=np.fft.ifftshift(Mminous)
        Fourier_minous=pyfftw.interfaces.numpy_fft.fft(M_minous)
        Fourier_plus=pyfftw.interfaces.numpy_fft.fft(M_plus)
        return Fourier_minous/cls.N_size, Fourier_plus/cls.N_size


    @classmethod
    def Get_Bands_Matrix_cluster(cls,Ground:bool =False)-> np.ndarray:
        Mminous, Mplus = cls.Sample_State(Ground=Ground)
        x=np.arange(-(cls.N_size-1)/2,(cls.N_size-1)/2+ 1)
        M_plus=pyfftw.empty_aligned(cls.N_size, dtype='complex128')
        M_plus[:]=np.fft.ifftshift(Mplus)
        M_minous[:]=np.fft.ifftshift(Mminous)
        Fourier_minous=np.fft.fft(M_minous)
        Fourier_plus=np.fft.fft(M_plus)
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
    def Covariance_matrix(cls,L:np.int64,Ground:bool=False)-> np.ndarray:
        Fourier_minous,Fourier_plus=cls.Get_Bands_Matrix(Ground=Ground)
        return (cls.Toeplitz_matrix(Fourier_plus,L)+cls.Hankel_matrix(Fourier_minous,L))
    @classmethod
    def Covariance_matrix_from_sub_sample(cls,Fourier_minous:np.ndarray,Fourier_plus:np.ndarray,L:np.int64)-> np.ndarray:
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
    @classmethod
    def Binary_entropy(cls,x:np.ndarray)->np.ndarray:
        result=[0 if np.abs(i-1)<10E-12 or np.abs(i)<10E-12 else -i*np.log(i)-(1-i)*np.log(1-i) for i in x]
        return np.array(result)


class Computations_XY_model(Sampling_Random_State):

    beta = np.min(Sampling_Random_State.Omega(np.linspace(-np.pi,np.pi,int(1000))))

    @classmethod
    def Sample_Fermi_dirac(cls,n:np.int64,Size:np.int64) -> np.float64:
        # beta is the inverse thermic energy associated in the system (beta)
        # mu corresponds to the chemical potential
        # n is the position of the particle
        # f=np.exp(T*(Omega(Gamma,Lambda,2.0*(np.pi/N)*n)-mu)) +1
        # N corresponds to the size of the system
        f=np.exp(cls.beta*(cls.Omega(((2.*np.pi)/np.float64(Size)) * n)-cls.mu)) +1
        return 1/f
    @classmethod
    def Compute_Entropy_State(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,n_size:np.int64=100,step:np.int64=2)->np.ndarray:
        """
        This function computes the Entropy of a given state being this the reason why the Fourier plus and the minous band have
        to be passed as a parameters. by default we compute the entropy for a size of 2 up to 100 and therfore we return
        an array.
        """
        S = [np.sum(cls.Binary_entropy(0.5-np.linalg.svd(cls.Covariance_matrix_from_sub_sample(Fourier_minous=Fourier_M, Fourier_plus=Fourier_P, L=i),compute_uv=False))) for i in range(2,n_size,step)]
        return S

    @classmethod
    def Compute_Density_Matrix_Random_State(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,L:np.int64)->np.ndarray:
        """
        This function returns the  density matrix from a random state, this is why we need to pass the Fourier plus and minous
        to this function, this does not compute the fourier transform, only the density matrix associated with it.
        """
        O_1, S, O_2 = np.linalg.svd(cls.Covariance_matrix_from_sub_sample( Fourier_minous=Fourier_M,Fourier_plus=Fourier_P, L=L))
        S = -S +0.5
        x= np.log(1-S) - np.log(S)
        M = -(O_1@np.diag(x)@O_2)/cls.beta
        return M
    @classmethod
    def Compute_Spectrum_Random_Distribution_Associated(cls,Fourier_M:np.ndarray,Fourier_P:np.ndarray,L:np.int64)->np.ndarray:
        S = np.linalg.svd(cls.Covariance_matrix_from_sub_sample(Fourier_minous=Fourier_M,Fourier_plus= Fourier_P,L=L),compute_uv=False)
        n=np.arange(-(L-1)/2,(L-1)/2 +1)
        S=sorted(-S+0.5,reverse=True)
        Fermi = sorted(cls.Sample_Fermi_dirac(n=n,Size=L),reverse=True)
        return S,Fermi
    @classmethod
    def Compute_Fourier_Transforms(cls,Ground = False,Save=False,Route = None,Cluster = False):
        if Cluster:
            Fourier_minous = np.zeros((cls.num_data,cls.N_size))
            Fourier_plus = np.zeros((cls.num_data,cls.N_size))
            for i in range(cls.num_data):
                a,b = cls.Get_Bands_Matrix_cluster(Ground=Ground)
                Fourier_minous[i,:]=a.real
                Fourier_plus[i,:]=b.real
        else:
            Fourier_minous = np.zeros((cls.num_data,cls.N_size))
            Fourier_plus = np.zeros((cls.num_data,cls.N_size))
            for i in range(cls.num_data):
                a,b = cls.Get_Bands_Matrix_local(Ground=Ground)
                Fourier_minous[i,:]=a.real
                Fourier_plus[i,:]=b.real

        if Save:
            with open(Route + 'Fourier_plus.pkl','wb') as f:
                pickle.dump(Fourier_plus, f)
                f.close()
            with open(Route + 'Fourier_minous.pkl','wb') as f:
                pickle.dump(Fourier_minous, f)
                f.close()
        else:
            return Fourier_minous,Fourier_plus











# L=30
# State = Sampling_Random_State()
# F_minous,F_plus=State.Get_Bands_Matrix()
# beta = np.min(State.Omega(np.linspace(-np.pi,np.pi,int(1000))))
# New_cov_matrix=State.Covariance_matrix_from_sub_sample(F_plus,F_minous,L)
# S=np.linalg.svd(New_cov_matrix,compute_uv=False)
# n=np.arange(-(L-1)/2,(L-1)/2 +1)
# array_to_plot=sorted(-S+0.5,reverse=True)
# plt.plot(array_to_plot,label="Singular values")
# plt.plot(np.array(sorted(Fermi_dirac(n=n,Size=L,beta=beta),reverse=True)),label="Fermi distribution")
# plt.legend()
# plt.title("lenght of {}".format(L))
# plt.show()
