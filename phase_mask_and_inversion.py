####################################################################################
# Functions included in this script, in order: 
# get_liquid_condition(state)
# get_eutectic_condition(state)
# get_solid_condition(state)
# get_phase_masks(state)
# core_inversion_and_mixing(state, phase_masks)
# calculate_liquid_fraction(state, phase_masks)
# calculate_temperature(state, phase_masks)
# calculate_liquid_salinity(state, phase_masks)
# inversion(state)
####################################################################################


from classes import p, d
from misc_functions import calculate_liquid_density, calculate_eutectic_temperature
import numpy as np


####################################################################################
# Liquid Condition for Phase Mask, H_L
####################################################################################

def get_liquid_condition(state):
    """
    Condition for cell to be completely liquid

    H >= H_l = ρ_l (L_w + C_pl (T_L - T_m)) + ΔP 
    """
    # threshold at liquid condition
    C   = state.S / state.rho_l
    T_L = p.T_m - p.m * C - p.Gamma * (p.static_pressure + state.dP) 

    return state.rho_l * (p.L_w + p.C_pl * (T_L - p.T_m)) + state.dP
                    


####################################################################################
# Eutectic Condition for Phase Mask, H_E, and Inversion Formulae
####################################################################################

def get_eutectic_condition(state):
    """
    Condition for the eutectic front 

    H_E = ρ_l χ L_w + (a_0 + a_1 * χ) (T_E - T_m) + ΔP  and H >= H_s
    
    where a_0 = ρ_s C_ps and a_1 = ρ_l C_pl - ρ_s C_ps
    """
    a0 = p.rho_s * p.C_ps  
    a1 = state.rho_l * p.C_pl - p.rho_s * p.C_ps

    T_E = calculate_eutectic_temperature(state)
    chi_E = state.S / (state.rho_l * p.C_E)     # liquid fraction at eutectic threshold

    return state.rho_l * chi_E * p.L_w + (a0 + a1 * chi_E) * (T_E - p.T_m) + state.dP 



####################################################################################
# Solid Condition for Phase Mask, H_s
####################################################################################

def get_solid_condition(state):
    """
    Condition for a cell to be completely solid (eutectic)

    H_s = ρ_s C_ps (T_E - T_m) + ΔP
    """
    T_E = calculate_eutectic_temperature(state)

    return p.rho_s * p.C_ps * (T_E - p.T_m) + state.dP
 


####################################################################################
# Phase Mask
####################################################################################

def get_phase_masks(state):
    """
    Get phase masks for liquid core(L), mushy layer(M), eutectic front(E),and pure solid (S)

    core_mask: H >= H_l
    mushy_mask: (H < H_l) & (H >= H_E)
    eutectic_mask: (H < H_E) & (H >= H_s)
    solid_mask: H < H_s
    """
    H_l = get_liquid_condition(state)
    H_E = get_eutectic_condition(state)
    H_s = get_solid_condition(state)
    
    Liquid   = (state.H >= H_l)
    Mushy    = (state.H < H_l) & (state.H >= H_E)
    Eutectic = (state.H < H_E) & (state.H >= H_s)
    Solid    = (state.H < H_s)

    return Liquid, Mushy, Eutectic, Solid



####################################################################################
# Core Inversion and Mixing
####################################################################################

def core_inversion_and_mixing(state, phase_masks):
    """ 
    np.sum() the average enthalpy in the core, H_core, and bulk salinity, S_core. A 
    global value of the liquid density is also computd. The temperature in the core, 
    T_core, and the liquid salinity are found by inversion. The core is then mixed by 
    setting all temperature and salinity values to the inverted results. 
    """
    Liquid = phase_masks[0]

    V_core = np.sum(d.V[Liquid])   # Total volume of the core
    H_core = np.sum(state.H[Liquid] * d.V[Liquid]) / V_core
    S_core = np.sum(state.S[Liquid] * d.V[Liquid]) / V_core 

    rho_l_core = np.sum(state.rho_l[Liquid] * d.V[Liquid]) / V_core 
    T_core     = p.T_m + (H_core - state.dP - rho_l_core * p.L_w) / (rho_l_core * p.C_pl)
    C_core     = S_core / rho_l_core

    # Mixing 
    state.H[Liquid] = H_core
    state.S[Liquid] = S_core
    state.T[Liquid] = T_core
    state.C[Liquid] = C_core

    return



####################################################################################
# Inversion
####################################################################################

def calculate_liquid_fraction(state, phase_masks):
    """ 
    χ in core: χ = 1

    Formula for χ in mush: 
    aχ² + bχ + c = 0. 
    Therefore, 
    χ = (-b - sqrt(b² - 4ac)) / 2a, where 
    a = a1 Γ ΔP - ρ_l L_w, 
    b = (a1 m S) / ρ_l + a0 Γ ΔP + H - ΔP, 
    c = (a0 m S) / ρ_l,
    and
    a0 = ρ_s C_ps,
    a1 = ρ_l C_pl - ρ_s C_ps.
    Since "a" is always negative, the negative root is taken to ensure the value
    of χ is positive.   

    Formula for χ at the eutectic front:
    χ = (H - ΔP - a0 (T_E - T_m) ) / (ρ_l L_w + a1 (T_E - T_m) ), where 
    a0 = ρ_s C_ps, 
    a1 = ρ_l C_pl - ρ_s C_ps, 
    T_E = T_m - m C_E - Γ ΔP.
    
    χ in solid:
    χ = 0 
    """
    H, S, chi, dP, rho_l = state.H, state.S, state.chi, state.dP, state.rho_l

    Liquid, Mushy, Eutectic, Solid = phase_masks

    a0  = p.rho_s * p.C_ps  
    a1  = rho_l * p.C_pl - p.rho_s * p.C_ps
    a   = a1 * p.Gamma * (p.static_pressure + dP) - rho_l * p.L_w
    b   = (a1 * p.m * S) / rho_l + a0 * p.Gamma * (p.static_pressure + dP) + H - dP
    c   = (a0 * p.m * S) / rho_l
    T_E = calculate_eutectic_temperature(state)

    chi[Liquid]   = 1.0
    chi[Mushy]    = ((-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a))[Mushy]
    chi[Eutectic] = ((H - dP - a0 * (T_E - p.T_m)) / (rho_l * p.L_w + a1 * (T_E - p.T_m) ))[Eutectic]
    chi[Solid]    = 0.0

    return chi


def calculate_temperature(state, phase_masks):
    """ 
    Calculating the temperature of the mush, eutectic and solid.    
    Formula for temperature in mush: 
    T = T_m + (H - ΔP - ρ_l χ L_w) / (a0 + a1 χ)
    
    Temperature at the eutectic front:
    T = T_E, where T_E is obtained from calculate_eutectic_temperature()
    
    Formula for temperature in solid:
    T = T_m + (H - ΔP) / (ρ_s C_ps)  
    """
    H, chi, T, dP, rho_l = state.H, state.chi, state.T, state.dP, state.rho_l

    Liquid, Mushy, Eutectic, Solid = phase_masks

    a0 = p.rho_s * p.C_ps  
    a1 = rho_l[Mushy] * p.C_pl - p.rho_s * p.C_ps
    T_E = calculate_eutectic_temperature(state)

    T[Mushy]    = p.T_m + (H[Mushy] - dP - rho_l[Mushy] * chi[Mushy] * p.L_w) / (a0 + a1 * chi[Mushy])
    T[Eutectic] = T_E
    T[Solid]    = p.T_m + (H[Solid] - dP) / (p.rho_s * p.C_ps)

    return T


def calculate_liquid_salinity(state, phase_masks):  
    """ 
    Formula for C in mush: 
    C = S / (ρ_l χ) 

    C at the eutectic front:
    C = C_E, where C_E is a constant

    C in solid:
    C = 0, since all solid
    """
    S, chi, C, rho_l = state.S, state.chi, state.C, state.rho_l

    Liquid, Mushy, Eutectic, Solid = phase_masks

    rho_l = calculate_liquid_density(state)
  
    C[Mushy] = np.divide(S, rho_l * chi, out=np.zeros_like(S), where=chi > 0)[Mushy]
    C[Eutectic] = p.C_E 
    C[Solid] = 0.0

    return C


def inversion(state):
    """ 
    Function that is called in main.py to invert enthalpy, H, and 
    bulk salinity, S, to obtain the liquid fraction, χ, temperature, T, 
    and liquid salinity, C. core_inversion_and_mixing is called in this
    function after inversion to mix the core. 
    """
    phase_masks = get_phase_masks(state)

    chi = calculate_liquid_fraction(state, phase_masks)
    T   = calculate_temperature(state, phase_masks)
    C   = calculate_liquid_salinity(state, phase_masks)

    # Mixes the core
    core_inversion_and_mixing(state, phase_masks)

    return chi, T, C 