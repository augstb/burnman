from __future__ import absolute_import

import numpy as np
import scipy.optimize as opt
import warnings

from . import equation_of_state as eos
from . import birch_murnaghan as bm
from . import debye
from .. import constants
from ..utils.math import bracket

class Sjostrom(eos.EquationOfState):

    def grueneisen_parameter(self, P, T, V, params):
        """
        Returns voluminal grueneisen parameter [unitless] as a function of
        volume [m^3] (EQ 13, 14).
        """
        return self._debye_grueneisen_parameter(V/params['V_ref'],params)

    def volume(self, P, T, params):
        """
        Returns volume [m^3] as a function of pressure [Pa] and
        temperature [K]. Finds it with dichotomy method. Starts from cold
        curve, and adds Einstein model, and electronic contributions
        (EQ 2, 10, 17).
        """
        func = lambda V: self.pressure(T, V, params)-P
        try:
            sol = bracket(func, params['V_0'], 1.e-2*params['V_0'])
        except:
            raise ValueError(
                'Cannot find a volume, perhaps you are outside of the range\
                of validity for the equation of state?')
        # V = opt.brentq(func, sol[0], sol[1])
        V = opt.fsolve(func, params['V_0'])[0] # Other resolution algorithm
        return V

    def isothermal_bulk_modulus(self, P, T, V, params):
        """
        Returns isothermal Bulk modulus [Pa] as a function of pressure [Pa],
        temperature [K], and volume [m^3] (EQ 3, 11, 17).
        """
        K_T = self._bm_bulk_modulus(V, params)
        return K_T

    def molar_heat_capacity_v(self, P, T, V, params):
        """
        Returns molar heat capacity at constant volume [J/K/mol] as a function
        of pressure [Pa], temperature [K], and volume [m^3] (EQ 9, 17).
        Magnetic contribution is deduced from F_mag (EQ 19).
        """
        x = V/params['V_0']
        C_V = 0.
        return C_V

    def thermal_expansivity(self, P, T, V, params):
        """
        Returns thermal expansivity [1/K] as a function of pressure [Pa],
        temperature [K], and volume [m^3] (EQ 12, 17).
        """
        x = V/params['V_0']
        alpha = 0.
        return alpha

    def molar_heat_capacity_p(self, P, T, V, params):
        """
        Returns molar heat capacity at constant pressure [J/K/mol] as a
        function of pressure [Pa], temperature [K], and volume [m^3].
        Computed from molar heat capacity at constant volume, thermal
        expansivity, and isothermal Bulk modulus.
        """
        C_P = 0.
        return C_P

    def adiabatic_bulk_modulus(self, P, T, V, params):
        """
        Returns adiabatic Bulk modulus [Pa] as a function of pressure [Pa],
        temperature [K], and volume [m^3]. Computed from isothermal Bulk
        modulus, thermal expansivity, and thermal expansivity.
        """
        K_S = 0.
        return K_S

    def pressure(self, T, V, params):
        """
        Returns pressure [Pa] as a function of temperature [K] and volume[m^3]
        (EQ 2, 10, 17).
        """
        x = V/params['V_0']
        P = self._bm_birch_murnaghan(x, params)
        return P

    def gibbs_free_energy(self, P, T, V, params):
        """
        Returns the Gibbs free energy [J/mol] as a function of pressure [Pa],
        volume [m^3] and temperature [K] of the mineral.
        """
        F = self.helmholtz_free_energy(P, T, V, params)
        G = F+P*V
        return G

    def molar_internal_energy(self, P, T, V, params):
        """
        Returns the internal energy [J/mol] as a function of pressure [Pa],
        volume [m^3] and temperature [K] of the mineral.
        """
        F = self.helmholtz_free_energy(P, T, V, params)
        S = self.entropy(P, T, V, params)
        U = F+T*S
        return U

    def entropy(self, P, T, V, params):
        """
        Returns the entropy [J/K/mol] as a function of pressure [Pa],
        volume [m^3] and temperature [K] of the mineral (EQ 7, 17).
        Magnetic contribution is deduced from F_mag (EQ 19).
        """
        S = 0.
        return S

    def enthalpy(self, P, T, V, params):
        """
        Returns the enthalpy [J/mol] as a function of pressure [Pa],
        volume [m^3] and temperature [K] of the mineral.
        """
        F = self.helmholtz_free_energy(P, T, V, params)
        S = self.entropy(P, T, V, params)
        H = F+T*S+P*V
        return H

    def helmholtz_free_energy(self, P, T, V, params):
        """
        Returns the Helmholtz free energy [J/mol] as a function of
        pressure [Pa], volume [m^3] and temperature [K] of the mineral.
        """
        x = V/params['V_0']
        F = params['U_0']+\
            self._bm_molar_internal_energy(P, T, V, params)
        return F

###############################################################################

    def _bm_birch_murnaghan(self, x, params):
        """
        equation for the third order birch-murnaghan equation of state, returns
        pressure in the same units that are supplied for the reference bulk
        modulus (params['K_0'])
        """
        return 3. * params['K_0'] / 2. * (pow(1/x, 7. / 3.) - pow(1/x, 5. / 3.))

    def _bm_bulk_modulus(self, volume, params):
        """
        compute the bulk modulus as per the third order
        birch-murnaghan equation of state.  Returns bulk
        modulus in the same units as the reference bulk
        modulus.  Pressure must be in :math:`[Pa]`.
        """

        x = params['V_0'] / volume
        f = 0.5 * (pow(x, 2. / 3.) - 1.0)

        K = pow(1. + 2. * f, 5. / 2.) * (params['K_0'] + (3. * params['K_0'] * params['Kprime_0'] -
                                                          5 * params['K_0']) * f + 27. / 2. * (params['K_0'] * params['Kprime_0'] - 4. * params['K_0']) * f * f)
        return K
    
    def _bm_molar_internal_energy(self, pressure, temperature, volume, params):
        """
        Returns the internal energy :math:`\mathcal{E}` of the mineral. :math:`[J/mol]`
        """
        x = np.power(volume/params['V_0'], -1./3.)
        x2 = x*x
        x4 = x2*x2
        x6 = x4*x2
        x8 = x4*x4
    
        xi1 = 3.*(4. - params['Kprime_0'])/4.
    
        intPdV = (-9./2. * params['V_0'] * params['K_0'] *
                  ((xi1 + 1.)*(x4/4. - x2/2. + 1./4.) -
                   xi1*(x6/6. - x4/4. + 1./12.)))
    
        return - intPdV
    
    def _debye_grueneisen_parameter(self, x, params):
        """
        Computes the grueneisen parameter according to Altshuler form, from the
        parameter x = V/V_ref (molar volumes) (EQ 13).
        """
        if 1/x >= 0:
            gamma = params['gamma_inf'] + \
                    x * (2*params['gamma_ref'] - 2*params['gamma_inf'] + params['gammaprime_R']) + \
                    x**2 * (params['gamma_inf'] - params['gamma_ref'] - params['gammaprime_R'])
        else:
            gamma = params['gamma_0'] + \
                    1/x * (2*params['gamma_ref'] - 2*params['gamma_0'] - params['gammaprime_L']) + \
                    (1/x)**2 * (params['gamma_0'] - params['gamma_ref'] + params['gammaprime_L'])
        return gamma
    
    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """
        if 'U_0' not in params:
            params['U_0'] = 0.

        # Now check all the required keys for the
        # thermal part of the EoS are in the dictionary
        expected_keys = ['V_0', 'K_0', 'Kprime_0',\
                         'theta_ref', 'gamma_ref', 'gammaprime_R', 'gammaprime_L', 'gamma_0', 'gamma_inf',
                         'V_ref', 'molar_mass', 'n']
        
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : '+k)

        # Finally, check that the values are reasonable.
        if params['molar_mass'] < 0.001 or params['molar_mass'] > 1.:
            warnings.warn('Unusual value for molar_mass', stacklevel = 2)
        if params['n'] < 1. or params['n'] > 1000.:
            warnings.warn('Unusual value for n', stacklevel = 2)
