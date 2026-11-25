####################################################################################
# Main script
####################################################################################
from classes import p, d, state
from functions import (
                    calculate_enthalpy_update,
                    calculate_bulk_salinity_update, 
                    calculate_overpressure_term, 
                    calculate_liquid_density,
                    calculate_liquid_enthalpy, 
                    calculate_ice_enthalpy, 
                    calculate_enthalpy,
                    calculate_bulk_salinity
                       )

from phase_mask_and_inversion import inversion


# Initial values
ice            = d.radial_centres > d.R_reservoir
state.chi[ice] = 0.0   
state.T[ice]   = p.T_cold

state.rho_l  = calculate_liquid_density(state)
state.h_l    = calculate_liquid_enthalpy(state)
state.h_s    = calculate_ice_enthalpy(state)
state.H      = calculate_enthalpy(state)
state.S      = calculate_bulk_salinity(state)
state.S[ice] = 0.0

# Iteration index        
i = 0

# Main loop
while i < d.N_iter:
   
    # Update enthalpy without overpressure term, and bulk salinity
    dH = calculate_enthalpy_update(state)
    dS = calculate_bulk_salinity_update(state)
    state.H += dH
    state.S += dS

    # Copies previous values for calculation of brine flux, J
    state.rho_l_old = state.rho_l.copy()
    state.chi_old   = state.chi.copy()

    # Inversion (including core mixing), lagged
    state.chi, state.T, state.C = inversion(state)  

    # Obtain dP using the new parameters from inversion(), and update 
    # the enthalpy accounting for the overpressure term
    dP_old = state.dP
    state.dP = calculate_overpressure_term(state)
    increase_in_dP = state.dP - dP_old 
    state.H += increase_in_dP
    
    # Fracture condition
    if state.dP > p.critical_overpressure:
        print(f"Critical overpressure = {state.dP / 1e6:.4f} MPa")
        t_fracture = (i + 1) * d.dt / (365.25 * 24 * 3600)
        print(f"Time to fracture = {t_fracture:.4f} years")
        print(f"Number of iterations required = {i}")
        break

    # Update the rest of the variables 
    state.rho_l = calculate_liquid_density(state)


    i += 1

# Printing key parameters
print(f"Time increment = {d.dt :.4f} s")
print(f"Cell size = {d.dr :.4f} m")

# Plot the temporal evolution animations and overpressure against time
