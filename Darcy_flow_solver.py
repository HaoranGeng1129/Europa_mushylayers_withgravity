####################################################################################
# Functions included in this script, in order: 
# calculate_permeability_at_centres(state)
# calculate_permeability_at_radial_faces(state)
# calculate_permeability_at_polar_faces(state)
# calculate_vertical_coordinates_at_centres()
# calculate_vertical_coordinates_at_radial_faces()
# calculate_vertical_coordinates_at_polar_faces()
# calculate_coefficient_in_mathcal_A(state)
# calculate_mathcal_A(mathcal_A_North, mathcal_A_South, mathcal_A_East, mathcal_A_West, mathcal_A_Centre)
# calculate_source_term(state, dP_old)
# calculate_gravity_terms(state, mathcal_A_North, mathcal_A_South, mathcal_A_East, mathcal_A_West)
# calculate_mathcal_A_and_mathcal_B(state, dP_old)
# calculate_perturbation_pressures(state, dP_old)
# calculate_flow_speed_at_faces(state)
# get_flow_direction_masks(state)
####################################################################################


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from classes import p, d


####################################################################################
# Permeabilites on the centres and faces, Pi
####################################################################################

def calculate_permeability_at_centres(state):
    """ 
    Define the permeability as
    Π_(i, j) = Π_0 χ^3. 

    Size: (N_radial, N_polar)
    """
    return p.Pi_0 * state.chi ** 3  


def calculate_permeability_at_radial_faces(state):
    """ 
    The permeability at the radial faces is taken as the harmonic mean of the 
    permeabilites of the adjacent cells, i.e. 
    1 / Π_(i + 1/2, j) = 1 / Π_(i, j) + 1 / Π_(i + 1, j).

    Alternatively, expressed as
    Π_(i + 1/2, j) = Π_(i, j) * Π_(i + 1, j) / (Π_(i, j) + Π_(i + 1, j)).

    Note that the permeabilities of the cells depend on the liquid fraction, 
    np.divide() is used to avoid division by zero in solid regions with Π = 0.

    Size: (N_radial - 1, N_polar)
    """
    Pi = calculate_permeability_at_centres(state)
    denominator = Pi[:-1, :] + Pi[1:, :]
    
    return np.divide(Pi[:-1, :] * Pi[1:, :], denominator, 
                    out=np.zeros_like(denominator), where=denominator > 0) 


def calculate_permeability_at_polar_faces(state):
    """ 
    The permeability at the polar faces, like radial, taken as harmonic mean 

    Π_(i, j + 1/2) = Π_(i, j) * Π_(i, j + 1) / (Π_(i, j) + Π_(i, j + 1)).

    Size: (N_radial, N_polar - 1)
    """
    Pi = calculate_permeability_at_centres(state)
    denominator = Pi[:, :-1] + Pi[:, 1:]
    
    return np.divide(Pi[:, :-1] * Pi[:, 1:], denominator, 
                    out=np.zeros_like(denominator), where=denominator > 0)



####################################################################################
# Vertical coordinates on the centres and faces, z
####################################################################################

def calculate_vertical_coordinates_at_centres(): 
    """ 
    In spherical coordinates, the vertical coordinates, z, is given by
    z = rcosθ.
    np.outer(), the outer product,  is used to populate a N_radial x N_polar matrix

    Size: (N_radial, N_polar)
    """
    return np.outer(d.radial_centres, np.cos(d.polar_centres))


def calculate_vertical_coordinates_at_radial_faces():
    """ 
    z = r * cosθ, where θ is held constant

    Size: (N_radial + 1, N_polar)
    """
    return np.outer(d.radial_faces, np.cos(d.polar_centres)) 


def calculate_vertical_coordinates_at_polar_faces():
    """ 
    z = r * cosθ, with r held constant

    Size: (N_radial, N_polar + 1)
    """
    return np.outer(d.radial_centres, np.cos(d.polar_faces)) 



####################################################################################
# Terms in mathcal_A (coefficient matrix)
####################################################################################

def calculate_coefficient_in_mathcal_A(state):
    """ 
    Calculating the coefficients and returning them separately, since these results
    appear again in mathcal_B. 
    A_North, the coefficient of cell at (i + 1/2, j); 
    A_South, the coefficient of cell at (i - 1/2, j);
    A_East, the coefficient of cell at (i, j + 1/2);
    A_West, the coefficient of cell at (i, j - 1/2).    
    np.concatenate() is used on empty rows and columns so that the results are of the 
    same size. 
    """ 
    state.Pi_radial_faces = calculate_permeability_at_radial_faces(state)
    state.Pi_polar_faces  = calculate_permeability_at_polar_faces(state)

    mathcal_A_radial = d.A_radial_faces[1:-1, :] * state.Pi_radial_faces / d.dr   # (N_radial - 1, N_polar)
    mathcal_A_polar  = d.A_polar_faces[:, 1:-1] * state.Pi_polar_faces / (
                        d.radial_centres[:, None] * d.dtheta)   # (N_radial, N_polar - 1)

    mathcal_A_radial = np.concatenate(
        [d.row_with_zeros, mathcal_A_radial, d.row_with_zeros], axis=0)   # (N_radial + 1, N_polar)
    mathcal_A_polar  = np.concatenate(
        [d.column_with_zeros, mathcal_A_polar, d.column_with_zeros], axis=1)   # (N_radial, N_polar + 1)

    mathcal_A_North  = mathcal_A_radial[1:, :]     # (N_radial, N_polar)
    mathcal_A_South  = mathcal_A_radial[:-1, :]    # (N_radial, N_polar)
    mathcal_A_East   = mathcal_A_polar[:, 1:]      # (N_radial, N_polar)
    mathcal_A_West   = mathcal_A_polar[:, :-1]     # (N_radial, N_polar)
    mathcal_A_Centre = - mathcal_A_North - mathcal_A_South - mathcal_A_East - mathcal_A_West

    return mathcal_A_North, mathcal_A_South, mathcal_A_East, mathcal_A_West, mathcal_A_Centre


def calculate_mathcal_A(mathcal_A_North, mathcal_A_South, mathcal_A_East, mathcal_A_West, mathcal_A_Centre):
    """  
    Populating the mathcal_A matrix using sp.sparse.diagonal. Once np.ravel() is applied to each of 
    mathcal_A_North, mathcal_A_South, etc, they are inserted as diagonal entries into the matrix, mathcal_A.

    A_South corresponds to the A_{i - 1, j} entry, offset by -N_polar;
    A_West corresponds to the A_{i, j - 1} entry, offset by -1;
    A_Centre corresponds to the A_{i, j} entry, with zero offset;
    A_East corresponds to the A_{i, j + 1} entry, offset by +1;
    A_North corresponds to the A_{i + 1, j} entry, offset by +N_polar.

    A_South and A_West are trimmed from the front, since the cells with the smallest radii and smallest polar angle
    do not have any southward or westward facing neighbours. In contrast, A_East and A_North are trimmed from the 
    back, since the cells with largest radii and polar angles do not have any northward or eastward facing facing
    neighbours. 

    [-d.N_polar, -1, 0, 1, d.N_polar] represents the offsets of each diagonal entry. 
    """
    mathcal_A = sp.diags([mathcal_A_South.ravel()[d.N_polar:],    # offset by -N_polar, length N_radial * N_polar - N_polar
                          mathcal_A_West.ravel()[1:],             # offset by -1, length N_radial * N_polar - 1
                          mathcal_A_Centre.ravel(),               # offset by 0, length N_radial * N_polar
                          mathcal_A_East.ravel()[:-1],            # offset by +1, length N_radial * N_polar - 1
                          mathcal_A_North.ravel()[:-d.N_polar]],  # offset by N_polar, length N_radial * N_polar - N_polar
                          [-d.N_polar, -1, 0, 1, d.N_polar],      # array that holds the offsets
                          format='csc'                            # sp.spsolve() requires matrices in CSC format
    )

    return mathcal_A



####################################################################################
# Terms in Curly_B, i.e. source term and gravity terms
####################################################################################

def calculate_source_term(state, dP_old):
    """ 
    souce, the source term, defined in Week 7 Progress
    B = source + gravity_terms

    Size: (N_radial, N_polar)
    """
    change_in_dP_term  = p.beta * state.chi * (state.dP - dP_old) / d.dt
    change_in_chi_term = (p.rho_s - state.rho_l) * (state.chi - state.chi_old) / (state.rho_l * d.dt)

    source = p.eta * d.V * (change_in_dP_term - change_in_chi_term) / 2

    return source.ravel()


def calculate_gravity_terms(state, mathcal_A_North, mathcal_A_South, mathcal_A_East, mathcal_A_West):
    """ 
    All gravity terms are defined in Week 7 Progress:
    B = source - B_North + B_South - B_East + B_West.

    B_North, the gravity term from cell at (i + 1/2, j); 
    B_South, the gravity term from cell at (i - 1/2, j);
    B_East, the gravity term from cell at (i, j + 1/2);
    B_West, the gravity term from cell at (i, j - 1/2).

    The arrays with labels "centre_minus_face" and "face_minus_centre" are created to save computing 
    time. np.concatenate() is called again to make sure that mathcal_B is defined for every cell. 

    Size: (N_radial, N_polar)
    """
    rho_l = state.rho_l     # (N_radial, N_polar)

    z_centres      = calculate_vertical_coordinates_at_centres()        # (N_radial, N_polar)
    z_radial_faces = calculate_vertical_coordinates_at_radial_faces()   # (N_radial + 1, N_polar)
    z_polar_faces  = calculate_vertical_coordinates_at_polar_faces()    # (N_radial, N_polar + 1)

    B_radial_centre_minus_face = (
        rho_l[1:, :] - p.rho_0) * p.g * (z_centres[1:, :] - z_radial_faces[1:-1, :]
    )  # (N_radial - 1, N_polar)
    B_radial_face_minus_centre = (
        rho_l[:-1, :] - p.rho_0) * p.g * (z_radial_faces[1:-1, :] - z_centres[:-1, :]
    ) # (N_radial - 1, N_polar)
    B_polar_centre_minus_face = (
        rho_l[:, 1:] - p.rho_0) * p.g * (z_centres[:, 1:] - z_polar_faces[:, 1:-1]
    )   # (N_radial, N_polar - 1)
    B_polar_face_minus_centre = (
        rho_l[:, :-1] - p.rho_0) * p.g * (z_polar_faces[:, 1:-1] - z_centres[:, :-1]
    )  # (N_radial, N_polar - 1)

    B_radial_no_boundaries = B_radial_centre_minus_face + B_radial_face_minus_centre     
    B_polar_no_boundaries  = B_polar_centre_minus_face + B_polar_face_minus_centre      

    B_radial = np.concatenate(
        [d.row_with_zeros, B_radial_no_boundaries, d.row_with_zeros], axis=0
    )     # (N_radial + 1, N_polar)
    B_polar  = np.concatenate(
        [d.column_with_zeros, B_polar_no_boundaries, d.column_with_zeros], axis=1
    )     # (N_radial, N_polar + 1)

    B_North  = mathcal_A_North * B_radial[1:, :]     # (N_radial, N_polar)
    B_South  = mathcal_A_South * B_radial[:-1, :]    # (N_radial, N_polar)
    B_East  = mathcal_A_East * B_polar[:, 1:]        # (N_radial, N_polar)
    B_West  = mathcal_A_West * B_polar[:, :-1]       # (N_radial, N_polar)      

    gravity_term = (- B_North + B_South - B_East + B_West).ravel()

    return B_radial_no_boundaries, B_polar_no_boundaries, gravity_term 



####################################################################################
# Matrix solve to determine the perturbation pressures
####################################################################################

def calculate_mathcal_A_and_mathcal_B(state, dP_old): 
    """ 
    In mathcal_A P' = mathcal_B
    A_Centre[A_Centre == 0] = 1.0 is needed because mathcal_A is singular if the cell 
    is solid (the row in mathcal_A that corresponds to said cell is all zeros). This is 
    because the permeabilities are all zero. Imposing A_Centre = 1.0 sets P' to the 
    value of the source term with no fluxes. 
    """
    # mathcal A
    A_North, A_South, A_East, A_West, A_Centre = calculate_coefficient_in_mathcal_A(state)
    A_Centre[A_Centre == 0] = 1.0   
    mathcal_A = calculate_mathcal_A(A_North, A_South, A_East, A_West, A_Centre)

    # mathcal_B
    source = calculate_source_term(state, dP_old)
    B_radial_no_boundaries, B_polar_no_boundaries, gravity_terms = calculate_gravity_terms(
        state, A_North, A_South, A_East, A_West
    )

    mathcal_B = source + gravity_terms

    return mathcal_A, mathcal_B, B_radial_no_boundaries, B_polar_no_boundaries


def calculate_perturbation_pressures(mathcal_A, mathcal_B):
    """ 
    A linear solve, mathcal_A P' = mathcal_B, using np.spsolve
    """
    P_prime = spla.spsolve(mathcal_A, mathcal_B)

    return P_prime.reshape(d.N_radial, d.N_polar)



####################################################################################
# Darcy velocities
####################################################################################

def calculate_flow_speed_at_faces(state, dP_old):
    """ 
    From Darcy's law (radial), 
    u_radial_faces 
    Size: (N_radial + 1, N_polar)

    and from Darcy's law (polar), 
    u_polar_faces 
    Size: (N_radial, N_polar + 1)
    
    """
    mathcal_A, mathcal_B, B_radial_no_boundaries, B_polar_no_boundaries = calculate_mathcal_A_and_mathcal_B(state, dP_old)

    P_prime = calculate_perturbation_pressures(mathcal_A, mathcal_B)

    radial_term_in_brackets = P_prime[1:, :] - P_prime[:-1, :] + B_radial_no_boundaries
    polar_term_in_brackets  = P_prime[:, 1:] - P_prime[:, :-1] + B_polar_no_boundaries

    # Of sizes (N_radial - 1, N_polar) and (N_radial, N_polar - 1), respectively
    u_radial_faces = - 2 * state.Pi_radial_faces * radial_term_in_brackets/ (p.eta * d.dr)
    u_polar_faces  = - 2 * state.Pi_polar_faces * polar_term_in_brackets/ (
                    p.eta * d.radial_centres[:, None] * d.dtheta)

    # Imposing zero flow boundary conditions
    u_radial_faces = np.concatenate([d.row_with_zeros, u_radial_faces, d.row_with_zeros], axis=0)
    u_polar_faces  = np.concatenate([d.column_with_zeros, u_polar_faces, d.column_with_zeros], axis=1)

    return u_radial_faces, u_polar_faces
