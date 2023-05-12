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

y = 0 # Counter to iterate over different beta for different plots
COM = np.zeros([1000,3])
# Loop to cycle through various T_demon values for plotting
for j in [0.1,0.2,0.5]:
    omega = 1 # Constant for hamiltonian
    beta = j # Inverse temperature constant
    T_demon = 0.1 # Demon temperature
    
    # The hamiltonian of the qubit
    H_1 = omega * (np.array([[1,0],[0,0]]))

    # The inital state of the system at an inverse temperature beta
    Z = np.trace(expm(-H_1*beta))
    rho = expm(-H_1*beta)/Z

    kappa_range = np.linspace(0,1,1000)
    x = 0 # Counter to store efficency values
    #Loop through measurement strength parameter, kappa
    for i in kappa_range:
        kappa = i 
    
        # Perform measurement
        rho_m = (M_plus(kappa) @ rho @ np.conjugate(M_plus(kappa)).T) / np.trace(M_plus(kappa) @ rho @ np.conjugate(M_plus(kappa)).T)
        Q_m = Energy(rho_m, H_1) - Energy(rho, H_1) #Energy added by measurement
    
        # Feedback stroke
        z_vector_len = np.sqrt(sum(np.array(BlochCoords(rho_m))**2)) #Find Bloch vector length
        rho_fb = (I - (z_vector_len * sigma_z))/2 #Rotate bloch vector to be on z-axis
        
        # Energy values
        W_ext = Energy(rho_m,H_1) - Energy(rho_fb, H_1) #Work done on system
        Q_m = Energy(rho_m,H_1) - Energy(rho,H_1) # Heat added through measurement
        Q_therm = Energy(rho,H_1) - Energy(rho_fb,H_1) 
        
        # Erasure work
        W_eras = -T_demon * (Entropy(rho_fb)-Entropy(rho))
        # Constant erasure term:
        #W_eras = T_demon * np.log(2)
        
        # Calculate efficency
        COM[x,y] = Q_therm / (W_ext + W_eras)
        x+=1
    y+=1

#%% Plotting
plt.plot(kappa_range,COM[:,0], linestyle='--', color='steelblue', linewidth=2, dashes=(10, 2), label = r'$k_B\beta$ = 0.1 $K^{-1}$')
plt.plot(kappa_range,COM[:,1], linestyle='-.', color='tomato', linewidth=2, dashes=(10,2,3,2), label = r'$k_B\beta$ = 0.2 $K^{-1}$')
plt.plot(kappa_range,COM[:,2], linestyle=':', color='k', linewidth=2, dashes=(2, 1), label = r'$k_B\beta$ = 0.5 $K^{-1}$')

plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('$C$',fontsize=18)
plt.xlabel('$\kappa$',fontsize=18)
plt.legend(fontsize=12)

#plt.savefig('COM vs. kappa (modified Launders erasure).pdf', bbox_inches='tight')
