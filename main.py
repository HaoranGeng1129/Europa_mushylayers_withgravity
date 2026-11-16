####################################################################################
# Main script
####################################################################################
import numpy as np
from classes import p, d, state, h
from functions import ( 

                       )

from phase_mask_and_inversion import inversion
from Animation import plot_profiles

# Initial values


# Iteration index        
i = 0

# Main loop
while i < d.N_iter:

    # Update fluxes
    
   
    # Update enthalpy without overpressure term and bulk salinity
    H_flux = state.J * h_faces + d.A * state.q   # enthalpic flux at the faces
    S_flux = state.J * C_faces
    state.H -= d.dt * (H_flux[1: ] - H_flux[: -1]) / d.V
    state.S -= d.dt * (S_flux[1: ] - S_flux[: -1]) / d.V

    # Copies previous values for calculation of brine flux, J
    state.rho_l_old = state.rho_l.copy()
    state.chi_old   = state.chi.copy()

    # Inversion
    state.chi, state.T, state.C = inversion(state)  

    # Obtain dP using the new parameters from inversion(), and update 
    # the enthalpy accounting for the overpressure term
    dP_old = state.dP
    state.dP = calculate_overpressure_term(state, M_0, d.V)
    increase_in_dP = state.dP - dP_old 
    
    state.H += increase_in_dP

    # Fracture condition
    if state.dP > p.critical_overpressure:
        print(f"Critical overpressure = {state.dP / 1e6:.4f} MPa")
        t_fracture = (i + 1) * d.dt / (365.25 * 24 * 3600)
        print(f"Time to fracture = {t_fracture:.4f} years")
        print(f"Number of iterations required = {i}")
        break

    # Calculate lagged parameters 
    state.rho_l = calculate_liquid_density(state)
    state.J = calculate_brine_flux(state, d.V, d.dt) 
    state.u_r = calculate_radial_speed(state, d.r_f)

    # Store historic data
    if i % 500 == 0:
            h.T_history[i // 500, :]     = state.T  
            h.rho_l_history[i // 500, :] = np.where(state.chi > 0, state.rho_l, 0.0)  
            h.H_history[i // 500, :]     = state.H     
            h.S_history[i // 500, :]     = state.S / 1000     
            h.dP_history[i // 500]       = state.dP    
            h.time_history[i // 500]     = (i + 1) * d.dt / (365.25 * 24 * 3600) 

    i += 1

# Printing key parameters
print(f"Time increment = {d.dt :.4f} s")
print(f"Cell size = {d.dr :.4f} m")

# Plot the temporal evolution animations and overpressure against time
plot_profiles()