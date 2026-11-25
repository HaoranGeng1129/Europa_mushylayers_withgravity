import numpy as np
from classes import p, d

####################################################################################
# Overpressure Term, dP
####################################################################################

def calculate_overpressure_term(state):
    """ 
    dP = 1/β ln[(M_0 - ρ_s Σ_i (1 - χ) V_i) / (ρ_0 Σ_i exp{β_S(C - C_0) χ V_i })]
    """
    reservoir, shell = get_reservoir_geometry_mask()

    numerator = p.M_reservoir - p.rho_s * np.sum( ((1 - state.chi) * d.V)[reservoir] )
    denominator = p.rho_0 * np.sum( 
        (np.exp(p.beta_S * (state.C - p.C_0)) * state.chi * d.V)[reservoir] )

    return np.log(numerator / denominator) / p.beta 



####################################################################################
# MISC
####################################################################################

def calculate_liquid_density(state):
    """ 
    EOS to calculate the density of liquid water

    ρ = ρ_0 exp[β ΔP + β_S (C - C_0)].
    """ 
    return p.rho_0 * np.exp(p.beta * state.dP + p.beta_S * (state.C - p.C_0))


def calculate_liquidus_temperature(state):
    """
    Liquidus temperature of briny cryomagma

    T_L = T_m - m C - Γ (P_0 + ΔP) 
    """
    return p.T_m - p.m * state.C - p.Gamma * (p.static_pressure + state.dP)


def calculate_eutectic_temperature(state):
    """
    Eutectic temperature of briny cryomagma

    T_E = T_m - m C_E - Γ (P_0 + ΔP)    
    """
    return p.T_m - p.m * p.C_E - p.Gamma * (p.static_pressure + state.dP)


def get_reservoir_geometry_mask():
    """ 
    Geometry mask to differentiate the reservoir (0 < r < R_reservoir), and the 
    surrounding heat-conducting shell (R_reservoir < r < R_domain). Note that the
    mask has to be of dimension N_radial x N_polar. Call np.broadcast_to() to 
    broadcast a row vector of length N_radial down N_polar columns
    """

    reservoir = np.broadcast_to((d.radial_centres <= d.R_reservoir)[:, None], (d.N_radial, d.N_polar))
    shell     = np.broadcast_to((d.radial_centres >  d.R_reservoir)[:, None], (d.N_radial, d.N_polar))

    return reservoir, shell 