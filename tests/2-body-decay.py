import numpy as np
import vector

#parent and daughter particle masses (in MeV/c^2)
mp = 139.6 #pion
m1 = 105.7 #muon
m2 = 0.12*1e-6 #muon neutrino is less than this

#calculate magnitude of momentum of daughter particles in COM frame
p = np.sqrt(m1**4+(m2**2-mp**2)**2-(2*(m1**2)*(m2**2+mp**2)))/(2*mp)

#number of simulations
N = 1000

#sampling angles
cos_theta = np.random.uniform(-1,1,N)
phi = np.random.uniform(0,2*np.pi,N)
theta = np.arccos(cos_theta)

#spatial momentum components of daughter particles
px = p*np.sin(theta)*np.cos(phi)
py = p*np.sin(theta)*np.sin(phi)
pz = p*cos_theta

#energies of daughter particles
E1 = np.sqrt(m1**2+p**2)
E2 = np.sqrt(m2**2+p**2)

#daughter particle 4-vectors
daughter1 = vector.array(
    {
        "E": np.full(N, E1),
        "px": px,
        "py": py,
        "pz": pz,
    }
)

daughter2 = vector.array(
    {
        "E": np.full(N, E2),
        "px": -px,
        "py": -py,
        "pz": -pz,
    }
)

print("Daughter 1:", daughter1[0:3])
print("Daughter 2:", daughter2[0:3])