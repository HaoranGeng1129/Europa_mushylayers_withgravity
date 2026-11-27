####################################################################################
# Functions included in this script, in order: 
# calculate_liquid_enthalpy(state)
# calculate_ice_enthalpy(state)
# calculate_enthalpy(state)
# calculate_bulk_salinity(state)
# calculate_liquid_density_at_faces(state)
# calculate_liquid_enthalpy_at_faces(state)
# calculate_flow_speed_at_faces(state)
# calculate_weighted_thermal_conductivity_at_faces(state)
# calculate_heat_flux(state)
# calculate_enthalpic_flux(state)
# calculate_enthalpy_update(state)
# calculate_liquid_salinity_at_faces(state)
# calculate_bulk_salinity_update(state)
####################################################################################


from classes import p, d
import numpy as np


####################################################################################
# Enthalpy
####################################################################################

def calculate_liquid_enthalpy(state):
    """
    Specific enthalpy of liquid water

    h_l = L_w + C_pl * (T - T_m) + dP / ρ_l
    """
    return p.L_w + p.C_pl * (state.T - p.T_m) + state.dP / state.rho_l


def calculate_ice_enthalpy(state):
    """
    Specific enthalpy of ice

    h_s = C_ps (T - T_m) + dP / ρ_s
    """
    return p.C_ps * (state.T - p.T_m) + state.dP / p.rho_s


def calculate_enthalpy(state):
    """
    Enthalpy

    H = ρ_l h_l χ + ρ_s h_s (1 - χ)
    """
    return state.rho_l * state.h_l * state.chi + p.rho_s * state.h_s * (1 - state.chi)



####################################################################################
# Bulk Salinity
####################################################################################

def calculate_bulk_salinity(state):
    """
    Bulk salinity

    S = ρ_l C χ + ρ_s C_s (1 - χ), with C_s = 0
    """
    return state.rho_l * state.C * state.chi



####################################################################################
# Enthalpy Update
####################################################################################

def calculate_liquid_density_at_faces(state):
    """
    
    """ 
    rho_l_radial_faces = 0
    rho_l_polar_faces  = 0

    return rho_l_radial_faces, rho_l_polar_faces


def calculate_liquid_enthalpy_at_faces(state):
    """ 
    
    """
    state.h_l = calculate_liquid_enthalpy(state)    

    h_l_radial_faces = 0 
    h_l_polar_faces  = 0 

    return h_l_radial_faces, h_l_polar_faces


def calculate_flow_speed_at_faces(state):
    """ 
    From Darcy's law (radial), 

    and from Darcy's law (polar), 
    
    """
    u_radial_faces = 0
    u_polar_faces  = 0

    return u_radial_faces, u_polar_faces


def calculate_weighted_thermal_conductivity_at_faces(state):
    """ 
    The thermal conductivity weighted by liquid fraction, used to calculate heat 
    flux. Note that the thermal conductivities at the polar faces need to be
    calculated separately. 
    
    k_bar_r = 2 k_a k_b / (k_a + k_b),

    where k = k_l χ + k_s (1 - χ). 
    """
    k = p.k_l * state.chi + p.k_s * (1 - state.chi)

    k_bar_radial_faces = 2 * k[1:, :] * k[:-1, :] / (k[1:, :] + k[:-1, :])
    k_bar_polar_faces  = 2 * k[:, 1:] * k[:, :-1] / (k[:, 1:] + k[:, :-1])

    return k_bar_radial_faces, k_bar_polar_faces


def calculate_heat_flux(state):
    """ 
    Heat flux through unit radial area of the faces (J s^{-1})

    q^r = - k_bar_r (T_{i+1, j} - T{i, j}) / dr,

    That through unit polar area 

    q^θ = - (k_bar_theta/r_c) (T_{i, j+1} - T_{i, j}) / dθ). 

    for faces sandwiched between cells. The flux at r = 0 should 
    be zero since energy is conserved at the centre of the sphere. 
    The flux at r = R is related to the difference between the 
    temperature at the boundary, T_B, and the temperature of the
    last cell, T[-1]. Similarly, the flux at θ = 0, π should both 
    be zero. 
    """
    k_bar_radial_faces, k_bar_polar_faces = calculate_weighted_thermal_conductivity_at_faces(state)

    q_radial_faces = np.zeros((d.N_radial + 1, d.N_polar))
    q_polar_faces  = np.zeros((d.N_radial, d.N_polar + 1))

    q_radial_faces[0, :]    = 0.0
    q_radial_faces[1:-1, :] = - k_bar_radial_faces * (state.T[1:, :] - state.T[:-1, :]) / d.dr 
    q_radial_faces[-1, :]   = - p.k_s * (p.T_B - state.T[-1, :]) / (d.dr / 2)

    q_polar_faces[:, 0]    = 0.0
    q_polar_faces[:, 1:-1] = - k_bar_polar_faces * (state.T[:, 1:] - state.T[:, :-1]
                                                    ) / (d.radial_centres[:, None] * d.dtheta)
    q_polar_faces[:, -1]   = 0.0

    return q_radial_faces, q_polar_faces 


def calculate_enthalpic_flux(state):
    """ 
    J_r = ρ_l h_l u_r + q_r, and

    J_theta = ρ_l h_l u_theta + q_theta.
    """
    rho_l_radial_faces, rho_l_polar_faces = calculate_liquid_density_at_faces(state)
    h_l_radial_faces, h_l_polar_faces     = calculate_liquid_enthalpy_at_faces(state)
    u_radial_faces, u_polar_faces         = calculate_flow_speed_at_faces(state)

    q_radial_faces, q_polar_faces = calculate_heat_flux(state)

    J_radial_faces = rho_l_radial_faces * h_l_radial_faces * u_radial_faces + q_radial_faces
    J_polar_faces  = rho_l_polar_faces * h_l_polar_faces * u_polar_faces + q_polar_faces

    return J_radial_faces, J_polar_faces


def calculate_enthalpy_update(state):
    """ 
    dH = dt (F_r{i+1/2, j} - F_r{i-1/2, j} + F_theta{i, j+1/2} - F_theta{i, j-1/2}) / V
    
    """
    J_radial_faces, J_polar_faces = calculate_enthalpic_flux(state)
    F_r     = d.A_radial_faces * J_radial_faces
    F_theta = d.A_polar_faces  * J_polar_faces 
    
    return - d.dt * (F_r[1:, :] - F_r[:-1, :] + F_theta[:, 1:] - F_theta[:, :-1])/ d.V





####################################################################################
# Bulk Salinity Update
####################################################################################

def calculate_liquid_salinity_at_faces(state):
    """ 
    
    """
    C_l_radial_faces = 0.0
    C_l_polar_faces  = 0.0

    return C_l_radial_faces, C_l_polar_faces


def calculate_bulk_salinity_update(state):
    """ 
    dS = dt (G_r{i+1/2, j} - G_r{i-1/2, j} + G_theta{i, j+1/2} - G_theta{i, j-1/2}) / V
    """
    rho_l_radial_faces, rho_l_polar_faces = calculate_liquid_density_at_faces(state)
    C_l_radial_faces, C_l_polar_faces     = calculate_liquid_salinity_at_faces(state)
    u_radial_faces, u_polar_faces         = calculate_flow_speed_at_faces(state)

    G_r     = d.A_radial_faces * rho_l_radial_faces * C_l_radial_faces * u_radial_faces
    G_theta = d.A_polar_faces  * rho_l_polar_faces * C_l_polar_faces * u_polar_faces

    return - d.dt * (G_r[1:, :] - G_r[:-1, :] + G_theta[:, 1:] - G_theta[:, :-1])/ d.V
