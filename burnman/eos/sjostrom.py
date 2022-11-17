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
        C_V = self._mag_molar_heat_capacity_v(T, params)
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
        alpha = self.thermal_expansivity(P, T, V, params)
        C_V = self.molar_heat_capacity_v(P, T, V, params)
        K_T = self.isothermal_bulk_modulus(P, T, V, params)
        C_P = C_V+alpha**2*T*V*K_T
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
        
        if(params['theta_ref']==0): thermal=0
        else: thermal=self._debye_pressure(T, V, params)
        
        x = V/params['V_0']
        P = self._bm_pressure(V, params)+\
            thermal
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
        S = self._mag_entropy(T, params)
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
        Debye_T = self._debye_temperature(V, params)
        
        if(params['theta_ref']==0): thermal=0
        else: thermal=debye.helmholtz_free_energy(T, Debye_T, params["n"])
        
        F = params['U_0']+\
            self._bm_molar_internal_energy(V, params)+\
            self._mag_helmholtz_free_energy(T, params)+\
            thermal
            
        return F

###############################################################################

    def _bm_pressure(self, volume, params):
        """
        equation for the fourth order birch-murnaghan equation of state, returns
        pressure in the same units that are supplied for the reference bulk
        modulus (params['K_0'])
        """
        eta = 0.5*((volume/params['V_0'])**(-2/3) - 1)
        deta = -1./(3 * params['V_0'])*(volume/params['V_0'])**(-5/3)
        factor = 9/2*params['K_0']*params['V_0']
        # a0 is prop included in U_0 : do not add the contribution here.
        a1 = factor*0
        a2 = factor*1
        a3 = factor*(params['Kprime_0'] - 4)
        a4 = factor*params['C_1']/np.math.factorial(4)
        return -a1*deta - 2*a2*eta*deta -3*a3*eta**2*deta -4*a4*eta**3*deta

    def _bm_bulk_modulus(self, volume, params):
        """
        compute the bulk modulus as per the third order
        birch-murnaghan equation of state.  Returns bulk
        modulus in the same units as the reference bulk
        modulus.  Pressure must be in :math:`[Pa]`.
        """
        eta = 0.5*((volume/params['V_0'])**(-2/3) - 1)
        deta = -1./(3 * params['V_0'])*(volume/params['V_0'])**(-5/3)
        d2eta = -5./(3 * params['V_0'])*(-1./(3 * params['V_0']))*(volume/params['V_0'])**(-8/3)
        factor = 9/2*params['K_0']*params['V_0']
        # a0 is prop included in U_0 : do not add the contribution here.
        a1 = factor*0
        a2 = factor*1
        a3 = factor*(params['Kprime_0'] - 4)
        a4 = factor*params['C_1']/np.math.factorial(4)
        return volume*(2*a2*eta*d2eta    + 2*a2*(deta)**2 + \
                       3*a3*eta**2*d2eta + 6*a3*eta*(deta)**2 + \
                       4*a4*eta**3*d2eta + 12*a4*eta**2*(deta)**2)
    
    def _bm_molar_internal_energy(self, volume, params):
        """
        Returns the internal energy :math:`\mathcal{E}` of the mineral. :math:`[J/mol]`
        """
        eta = 0.5*((volume/params['V_0'])**(-2/3) - 1)
        factor = 9/2*params['K_0']*params['V_0']
        # a0 is prop included in U_0 : do not add the contribution here.
        a1 = factor*0
        a2 = factor*1
        a3 = factor*(params['Kprime_0'] - 4)
        a4 = factor*params['C_1']/np.math.factorial(4)
        return a1*eta + a2*eta**2 + a3*eta**3 + a4*eta**4
    
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
    
    def _debye_temperature(self, volume, params):
        """
        Computes the Einstein temperature from the parameter x = V/V_0
        (molar volumes) (EQ 15).
        """
        rhoref = params['molar_mass']/params['V_ref']
        rho = params['molar_mass']/volume
        x=volume/params['V_ref']
        const = rhoref**params['gamma_inf']*np.exp(.5*(3*params['gamma_inf'] - params['gammaprime_R'] - 3*params['gamma_ref']))
        const = params['theta_ref']/const
        if 1/x >= 0:
            theta = rho**params['gamma_inf']*np.exp(rhoref/(2*rho**2)*(\
            4*params['gamma_inf']*rho - params['gamma_inf']*rhoref - 2*params['gammaprime_R']*rho +\
            params['gammaprime_R']*rhoref - 4*params['gamma_ref']*rho + params['gamma_ref']*rhoref))
        else:
            theta = rho**params['gamma_0']*np.exp(rho/(2*rhoref**2)*(\
            4*params['gamma_ref']*rhoref - params['gamma_ref']*rho - 2*params['gammaprime_L']*rhoref +\
            params['gammaprime_L']*rho - 4*params['gamma_0']*rhoref + params['gamma_0']*rho))
        
        return const*theta

    def _debye_pressure(self, T, V, params):
        Debye_T = self._debye_temperature(V, params)
        gr = self._debye_grueneisen_parameter(V/params['V_ref'], params)
        P_th = gr * debye.thermal_energy(T, Debye_T, params["n"]) / V
        return P_th

    def _mag_helmholtz_free_energy(self, T, params):
        """
        Computes the magnetic contribution to F from F_mag (EQ 11).
        """
        if params['T_c'] > 0 and T < params['T_c']:
            a = np.sqrt(1135.)
            b = 4680. / a**3
            rapport = (1+np.sqrt(T/a**2))/(1-np.sqrt(T/a**2))
            return a**3*b*( (1-T/a**2)*np.log(rapport) - 2*np.sqrt(T/a**2) + 4./3*(T/a**2)**3 )
        else: return 0.
    
    def _mag_entropy(self, T, params):
        """
        Computes the magnetic contribution to the entropy from F_mag (EQ 11).
        """
        if params['T_c'] > 0 and T < params['T_c']:
            a = np.sqrt(1135.)
            b = 4680. / a**3
            rapport = (1+np.sqrt(T/a**2))/(1-np.sqrt(T/a**2))
            return -4*b*T**2/a**3 + a*b*np.log(rapport)
        else: return 0.

    def _mag_molar_heat_capacity_v(self, T, params):
        """
        Computes the magnetic contribution to the molar heat capacity at
        constant volume, from F_mag (EQ 11).
        """
        if params['T_c'] > 0 and T < params['T_c']:
            a = np.sqrt(1135.)
            b = 4680. / a**3
            rapport = (1+np.sqrt(T/a**2))/(1-np.sqrt(T/a**2))
            return (-8*b*T**2 + 8*b*T**3/a**2 + a**4*b*np.sqrt(T/a**2))/(a**3*T - a*T**2)
        else: return 0.
        

    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """
        if 'U_0' not in params:
            params['U_0'] = 0.
        if 'C_1' not in params:
            params['C_1'] = 0.
        if 'T_c' not in params:
            params['T_c'] = 0.

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
