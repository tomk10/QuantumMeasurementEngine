import sympy as sym
import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm
import matplotlib.pyplot as plt


def Energy(rho,H):
    '''
    Returns the energy of a given density matrix for a given hamiltonian
    '''
    return np.trace(rho @ H)

# A measurement in the x direction, kappa idicates the strength
# kapppa = 1/2, no information gained,
# kappa = 0,1, maximum binary information gained
def M_plus(kappa):
    A = (np.sqrt(kappa) + np.sqrt(1-kappa)) * I
    B = (np.sqrt(kappa) - np.sqrt(1-kappa)) * sigma_x
    return 1/2*(A+B)

# Same measurement as above except in the minus direction
# sum(M_i ^2) = I
def M_minus(kappa):
    A = (np.sqrt(kappa) + np.sqrt(1-kappa)) * I
    B = (np.sqrt(kappa) - np.sqrt(1-kappa)) * sigma_x
    return 1/2*(A-B)

def BlochCoords(rho):
    '''
    Returns the Bloch sphere coordinates of the density matrix
    '''
    a = rho[0, 0]
    b = rho[1, 0]
    x = 2.0 * b.real
    y = 2.0 * b.imag
    z = 2.0 * a - 1.0
    return([x,y,z])

def Entropy(rho):
    return -np.trace(rho @ logm(rho))

I = np.array([[1,0],[0,1]]) # Identity matrix
#Pauli matrices:
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

Del_S_range = np.zeros([1000,3])
temp_range = np.linspace(0,80,1000)
x = 0 # Counter to store efficency values
#Loop through measurement strength parameter, kappa
for i in temp_range:
    beta = i # Inverse temperature constant
    omega = 0.1 # Constant for hamiltonian
    T_demon = 0.05
    kappa = 1

    # The hamiltonian of the qubit
    H_1 = omega * (np.array([[1,0],[0,0]]))

    # The inital state of the system at an inverse temperature beta
    Z = np.trace(expm(-H_1*beta))
    rho = expm(-H_1*beta)/Z

    # Perform measurement
    rho_m = (M_plus(kappa) @ rho @ np.conjugate(M_plus(kappa)).T) / np.trace(M_plus(kappa) @ rho @ np.conjugate(M_plus(kappa)).T)
    Q_m = Energy(rho_m, H_1) - Energy(rho, H_1) #Energy added by measurement

    # Feedback stroke
    z_vector_len = np.sqrt(sum(np.array(BlochCoords(rho_m))**2)) #Find Bloch vector length
    rho_fb = (I - (z_vector_len * sigma_z))/2 #Rotate bloch vector to be on z-axis
    
    # Erasure change in entropy
    Del_S = Entropy(rho)-Entropy(rho_m)
    Del_S_range[x] = Del_S
    x+=1

#%% Plotting
plt.plot(temp_range,Del_S_range[:,0], color='steelblue', linewidth=2, label = 'Qubit system')
plt.plot(temp_range,np.ones(len(temp_range)) * np.log(2), linestyle='--', color='g', linewidth= 2, label = r'$\Delta S = \ln(2)$')


plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('$\Delta$ S',fontsize=12)
plt.xlabel(r'$\beta$ ($K^{-1}$ $k_B^{-1}$)',fontsize=12)
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend()

#plt.savefig('Change in entropy vs. beta (modified Landauer's erasure).pdf',bbox_inches='tight')