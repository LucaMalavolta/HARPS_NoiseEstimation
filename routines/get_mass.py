import numpy as np
from scipy.optimize import fsolve

G_grav = 6.67428e-11 # Gravitational Constants in SI system [m^3/kg/s^2]
M_sun = 1.9884e30 # Value from TRADES

def get_mass(M_star2, M_star1, Period, K1, e0):
    # M_star1, M_star2 in solar masses
    # P in days -> Period is converted in seconds in the routine
    # inclination assumed to be 90 degrees
    # Gravitational constant is given in m^3 kg^-1 s^-2
    # output in m/s
    output = K1 - (2. * np.pi * G_grav * M_sun / 86400.0) ** (1.0 / 3.0) * (1.000 / np.sqrt(1.0 - e0 ** 2.0)) * (
                                                                                                                    Period) ** (
                                                                                                                    -1.0 / 3.0) * (
                      M_star2 * (M_star1 + M_star2) ** (-2.0 / 3.0))
    return output


def get_mass_from_K(M_star, P, K, e):
    x0 = 10.0/333000.0
    return fsolve(get_mass, x0, args=(M_star, P, K, e))*333000.0
