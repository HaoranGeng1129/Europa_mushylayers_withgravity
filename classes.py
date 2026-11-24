import numpy as np
from dataclasses import dataclass

####################################################################################
# Store Physical Parameters (frozen)
####################################################################################
# test change 
@dataclass(frozen=True)
class Physical_Parameters: 

    # Temperature related parameters
    T_m    : float = 273.15    # K, pure water freezing temperature
    T_cold : float = 222.0     # K, initial temperature of surrounding ice shell
    T_B    : float = 222.0     # K, temperature at the domain boundary
    T_0    : float = 275.0     # K, initial temperature of the liquid
    C_pl   : float = 3985.0    # Jkg^{-1}K^{-1}, brine specific heat capacity 
    C_ps   : float = 2100.0    # Jkg^{-1}K^{-1}, pure water ice specific heat capacity 
    L_w    : float = 3.335e5   # Jkg^{-1}, latent heat of fusion of water 
    k_l    : float = 0.6       # Wm^{-1}K^{-1}, liquid water thermal conductivity
    k_s    : float = 2.3       # Wm^{-1}K^{-1}, pure water ice thermal conductivity


    @property
    def alpha(self):
        """
        m^{2}s^{-1}, ice thermal diffusivity
        
        α = k_s / (ρ_0 C_pl)
        """
        return self.k_s / (self.rho_0 * self.C_pl)

    # Salt related parameters
    m      : float = 0.0228   # K (g/kg)^{-1}, freezing point depression coefficient
    C_0    : float = 0.0      # g/kg, reference salinity in EOS
    C_init : float = 35.0     # g/kg, initial salinity of the liquid
    beta_S : float = 8.3e-4   # (g/kg)^{-1}, saline contraction coefficient    
    
    # Water related parameters
    rho_0 : float = 1000.0    # kgm^{-3}, reference water density at 0 overpressure, salinity
    rho_s : float = 917.0     # kgm^{-3}, pure water ice
    eta   : float = 2.0e-3    # Pas, pure water dynamic viscosity, Buffo et al. (2020)

    @property
    def rho_b(self): 
        """ 
        kgm^{-3}, brine density used to compute Gamma
        
        ρ_b = ρ_0 exp{β_S (C_init - C_0)}
        """
        return self.rho_0 * np.exp(self.beta_S * (self.C_init - self.C_0) )

    # Eutectic parameters
    C_E : float = 175     # g/kg, liquid salinity at the eutectic point for MgSO4-H20

    @property
    def S_ice(self):
        """ 
        g/m^3, bulk salinity of the surrounding ice
        """
        return self.rho_b * self.C_init

    # Gravity relatd parameters
    g     : float = 1.315       # ms^{-2}, gravity on Europa, Pappalardo et al. (2009)
    depth : float = 2000.0      # m, depth of the resevoir, H in Lesage's paper

    @property 
    def M_reservoir(self): 
        """ 
        kg, total mass of the reservoir, a constant since only heat is exchanged at the
        reservoir boundary
        """
        return 4 * self.rho_b * np.pi * d.R_reservoir / 3

    # Pressure related parameters
    beta    : float = 3.5e-10   # Pa^{-1}, liquid water compressibility
    sigma_c : float = 1.5e6     # Pa , tensile strength of ice 

    @property
    def Gamma(self):
        """
        KPa^{-1}, pressure slope in liquidus condition, 
        
        Γ = T (1/ρ_s - 1/ρ_b) / L
        """
        return self.T_m * (1 / self.rho_s - 1 / self.rho_b) / self.L_w      

    @property
    def critical_overpressure(self): 
        """ 
        ΔP_c = 2(σ_c + ρ_s g H)
        """
        return 2 * (self.sigma_c + self.rho_s * self.g * self.depth)

    @property
    def static_pressure(self):
        """ 
        P_0 = ρ_s g H
        """
        return self.rho_s * self.g * self.depth

p = Physical_Parameters()



####################################################################################
# Store Domain Parameters
####################################################################################
class Domain_Parameters: 
    def __init__(self):
        self.N_iter   : int = 10**7              # number of iterations
        self.N_radial : int = 600                # number of cells (radial) 
        self.N_polar  : int = 64                 # number of cells (polar)

        self.R_reservoir : float = 500.0                # m, radius of the resevoir
        self.R_domain    : float = 600.0                # m, radius of the entire domain

        self.dr     : float = self.R_domain / self.N_radial  # m, radial distance increment
        self.dtheta : float = np.pi / self.N_polar           # rad, polar angle increment

        # Coordinates of the radial cell faces and centres, and polar faces and centres
        self.radial_faces   = np.linspace(0, self.R_domain, self.N_radial + 1)
        self.radial_centres = 0.5 * (self.radial_faces[1:] + self.radial_faces[:-1])              
        self.polar_faces    = np.linspace(0, np.pi, self.N_polar + 1)     
        self.polar_centres  = 0.5 * (self.polar_faces[1:] + self.polar_faces[:-1])     

        # Radial and polar areas, np.outer() to obtain an (N_r, N_theta) array
        self.A_radial_faces = 2 * np.pi * np.outer(self.radial_faces[:] ** 2, 
                                                         np.cos(self.polar_faces[:-1]) 
                                                         - np.cos(self.polar_faces[1:]))
        self.A_polar_faces  = np.pi * np.outer(self.radial_faces[1:] ** 2 - 
                                                            self.radial_faces[:-1] ** 2, 
                                                            np.sin(self.polar_faces[:]))

        # Volume
        self.V = 2 * np.pi * np.outer(self.radial_faces[1:] ** 3 - self.radial_faces[:-1] ** 3, 
                                np.cos(self.polar_faces[:-1]) - np.cos(self.polar_faces[1:])) / 3 
        
        self.dt : float = self.dr ** 2 /(2 * p.alpha * 10)  # s, time increment

d = Domain_Parameters()



####################################################################################
# Store States (not frozen)
####################################################################################
@dataclass(frozen=False)
class State: 
    H     : np.ndarray    # enthalpy
    h_l   : np.ndarray    # specific enthalpy of liquid brine
    h_s   : np.ndarray    # specific enthalpy of ice
    S     : np.ndarray    # bulk salinity 
    rho_l : np.ndarray    # density of the liquid
    T     : np.ndarray    # temperature 
    C     : np.ndarray    # liquid salinity 
    chi   : np.ndarray    # liquid fraction

    dP    : float         # overpressure term

    rho_l_old   : np.ndarray    # Store old liquid density values
    chi_old     : np.ndarray    # Store old liquid fraction values

# Defining dimensions of arrays
shape_centre       = (d.N_radial, d.N_polar)      # cell centre 
shape_radial_faces = (d.N_radial + 1, d.N_polar)  # radial faces
shape_polar_faces  = (d.N_radial, d.N_polar + 1)  # polar faces

state = State(
        H     = np.zeros(shape_centre), 
        h_l   = np.zeros(shape_centre), 
        h_s   = np.zeros(shape_centre), 
        S     = np.zeros(shape_centre), 
        rho_l = np.full(shape_centre, p.rho_b), 
        T     = np.full(shape_centre, p.T_0), 
        C     = np.full(shape_centre, p.C_init), 
        chi   = np.ones(shape_centre),
        dP    = 0.0, 
        rho_l_old = np.zeros(shape_centre), 
        chi_old   = np.zeros(shape_centre)
              ) 



####################################################################################
# Store Historic Data
####################################################################################
@dataclass(frozen=False)
class Historic_Data:
    T_history     : np.ndarray  # temperature 
    rho_l_history : np.ndarray  # liquid density
    H_history     : np.ndarray  # enthalpy
    S_history     : np.ndarray  # bulk salinity

    time_history  : np.ndarray
    dP_history    : np.ndarray  # overpressure term 


h = Historic_Data(
        T_history     = np.zeros((d.N_iter // 500, d.N)), 
        rho_l_history = np.zeros((d.N_iter // 500, d.N)), 
        H_history     = np.zeros((d.N_iter // 500, d.N)),  
        S_history     = np.zeros((d.N_iter // 500, d.N)),  

        time_history  = np.zeros(d.N_iter // 500),
        dP_history    = np.zeros(d.N_iter // 500)      
                )