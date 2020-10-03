import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time as timer

from numpy import fft
from scipy import linalg, integrate, interpolate
from matplotlib.animation import FuncAnimation


mpl.rc('font', size=14)
mpl.rc('text', usetex=True)
mpl.rc('font', family='sans-serif', serif='Times New Roman')

class Multi_Surface_SO(object):
    
    def __init__(self,grid_params):
        self.grid_params = grid_params
        self.hbar = 0.6582  # hbar in eV
        self.L0 = 0.19      # default diabatic coupling lambda0 in eV
        x_bc,p_bc = self.grid_params
        self.N = 2*p_bc + 1
        self.dx = 2*x_bc/self.N
        self.X = np.linspace(-x_bc,x_bc,self.N)
        
        self.P = (np.pi/x_bc)*np.arange(-p_bc,p_bc+1)
        self.time = 0.0
        
    def psi_x(self,inital_positions,initial_conditions):
        alpha,x10,p10,x20,p20 = initial_positions
        c1,c2 = initial_conditions
        psi_x1 = c1 * (alpha/np.pi)**(1/4) * np.exp(-alpha*(self.X-x10)**2 /2)* np.exp(1j*p10*(self.X-x10)/self.hbar)
        if c2!=0:
            psi_x2 = c2 * (alpha/np.pi)**(1/4) * np.exp(-alpha*(self.X-x10)**2 /2)* np.exp(1j*p20*(self.X-x10)/self.hbar)
        else:
            psi_x2 = np.empty(self.N)
            psi_x2.fill(0.000001)
        psi_x = np.concatenate((psi_x1,psi_x2))
        return psi_x
    
    def FFT(self,wfunc_x):
        psi_p  = self.dx/(2*np.pi) *fft.fftshift( fft.fft(wfunc_x))
        norm_const = np.trapz(np.abs(psi_p)**2,self.P)
        psi_p = 1/np.sqrt(norm_const) * psi_p
        return(psi_p)

    def iFFT(self,wfunc_p):
        psi_x  = (2*np.pi/self.dx)*fft.ifft(fft.ifftshift(wfunc_p),norm='ortho')
        norm_const = np.trapz(np.abs(psi_x)**2,self.X)
        psi_x = 1/np.sqrt(norm_const) * psi_x
        return(psi_x)
     
    def norm(self,wfunc):
        CX = np.concatenate((self.X,self.X))
        integrand  = np.abs(wfunc)**2
        norm = integrate.trapz(integrand,CX)
        return(norm)
    
    
    def expectation_value(self,func,wfunc,z):  
        # Takes a functional argument (func) and 
        # returns <func> using  give wave function (wfunc)
        integrand = func(z) * np.abs(wfunc)**2
        expec_val = integrate.trapz(integrand,z)
        return(expec_val)
        
    def Matrix_blocks(self,wave_packet_params, initial_positions):
        K1,K2,E0,m = wave_packet_params 
        alpha,x10,p10,x20,p20 = initial_positions
        V1 = 0.5*K1*(self.X - x10)**2
        self.V1 = np.diag(V1,k=0)
        
        V2 = E0 + 0.5*K2*(self.X - x20)**2
        self.V2 = np.diag(V2,k=0)
        
        T = (0.5/m) * self.P**2
        T = np.diag(T,k=0)
        self.T = np.kron(np.eye(2),T)
        
    def system_setup(self,wave_packet_params,initial_positions,laser_params,):
        self.Matrix_blocks(wave_packet_params,initial_positions)
        self.laser_params =  laser_params
        
    def Laser_coupling(self,t):
        g,tc,tw,freq,L0 = self.laser_params
        coupling = L0 #- g*np.exp(-(t-tc)**2 / tw**2)*np.cos(freq*(t-tc))
        coupling_matrix = coupling* np.eye(self.N)
        return(coupling_matrix)
    
    def Potential_matrix(self,t):
        V_int = self.Laser_coupling(t)
        #print(t)
        V = np.block([[self.V1,V_int],[V_int,self.V2]])
        return(V)
    
    def Propagator(self,chi1_ti,chi2_ti,dt):
        #print(chi1_ti,chi2_ti)
        self.time += dt
        #step 1,2
        Psi_P_ti = np.concatenate((self.FFT(chi1_ti),self.FFT(chi2_ti)))
        Psi_P_ti_p1 = np.dot(linalg.expm(-1j*dt/(2*self.hbar) * self.T), Psi_P_ti)
        #print(Psi_P_ti)
        #step 3,4
        Psi_X_ti_p1 = np.concatenate((self.iFFT(Psi_P_ti_p1[:self.N]), self.iFFT(Psi_P_ti_p1[self.N : 2*self.N])))
        
        V = self.Potential_matrix(self.time)
        Psi_X_ti_p2 = np.dot(linalg.expm(-1j*dt/(1*self.hbar)*V),Psi_X_ti_p1)
        
        #step 5,6
        Psi_P_ti_p2 = np.concatenate((self.FFT(Psi_X_ti_p2[:self.N]), self.FFT(Psi_X_ti_p2[self.N : 2*self.N]) ) )
        Psi_P_tj = np.dot(linalg.expm(-1j*dt/(2*self.hbar) *self.T), Psi_P_ti_p2)
        
        #step 7
        Psi_X_tj = np.concatenate((self.iFFT(Psi_P_tj[:self.N]), self.iFFT(Psi_P_tj[self.N : 2*self.N])) )
        Psi_X_tj = 1/self.norm(Psi_X_tj) * Psi_X_tj
        chi1_tj = Psi_X_tj[:self.N]
        chi2_tj = Psi_X_tj[self.N : 2*self.N]
        KE = np.linalg.multi_dot([np.conj(Psi_P_tj),self.T,Psi_P_tj.transpose()])
        PE = np.linalg.multi_dot([np.conj(Psi_X_tj),V,Psi_X_tj.transpose()])
        E= np.abs((KE,PE))
        return(chi1_tj,chi2_tj,E)
    
start_time = timer.time()

nSteps = 100
dt = 10
grid_params = (20,256) 
wave_packet_params = (0.02,0.02,0.44,86.65)  # K1,K2,E0,m
laser_params = (2.0,50,10,3.7678, 0.19)# g,tc,tw,freq,L0
initial_positions = (1,-8.67,0,5.62,0) # alpha,x10,p10,x20,p20
a =1 /np.sqrt(2)
initial_conditions = (a,a)  # c1,c2

surfs     = Multi_Surface_SO(grid_params)
surfs.system_setup(wave_packet_params,initial_positions,laser_params)
N = surfs.N
Psi_X0 = surfs.psi_x(initial_positions,initial_conditions)
chi1 = []
chi2 = []
E_check =[]
chi1_ti = Psi_X0[:N]
chi2_ti = Psi_X0[N:2*N]


for i in range(nSteps):
    chi1.append(chi1_ti)
    chi2.append(chi2_ti)
    E_check.append(E)
    chi1_ti,chi2_ti,E = surfs.Propagator(chi1_ti,chi2_ti,dt)

end_time = timer.time()

print( (end_time- start_time)/60 )
    

energy = [i[0]+i[1] for i in E_check ]
t = range(nSteps)
plt.plot(energy)
plt.title('Self Consistency Check')
plt.xlabel('t')
plt.ylabel('E(t)')