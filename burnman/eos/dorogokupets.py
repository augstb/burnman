from __future__ import absolute_import

import numpy as np
import scipy.optimize as opt
import warnings

from . import equation_of_state as eos
from .. import constants
from ..utils.math import bracket

class Dorogokupets(eos.EquationOfState):

    def grueneisen_parameter(self, P, T, V, params):
        """
        Returns voluminal grueneisen parameter [unitless] as a function of
        volume [m^3] (EQ 13, 14).
        """
        return self._grueneisen_parameter(V/params['V_0'],params)

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
        T_0 = params['T_0']
        K_T = self._vinet_bulk_modulus(V, params)+\
        self._lennard_jones_bulk_modulus(V, params)+\
        self._einstein_thermal_bulk_modulus(T, V, params)-\
        self._einstein_thermal_bulk_modulus(T_0, V, params)+\
        self._elec_thermal_bulk_modulus(T, V, params)-\
        self._elec_thermal_bulk_modulus(T_0, V, params)
        return K_T

    def molar_heat_capacity_v(self, P, T, V, params):
        """
        Returns molar heat capacity at constant volume [J/K/mol] as a function
        of pressure [Pa], temperature [K], and volume [m^3] (EQ 9, 17).
        Magnetic contribution is deduced from F_mag (EQ 19).
        """
        x = V/params['V_0']
        C_V = self._einstein_molar_heat_capacity_v(T, V, params)+\
        self._elec_molar_heat_capacity_v(T, x, params)+\
        self._mag_molar_heat_capacity_v(T, params)
        return C_V

    def thermal_expansivity(self, P, T, V, params):
        """
        Returns thermal expansivity [1/K] as a function of pressure [Pa],
        temperature [K], and volume [m^3] (EQ 12, 17).
        """
        x = V/params['V_0']
        C_V = self._einstein_molar_heat_capacity_v(T, V, params)
        gr = self._grueneisen_parameter(x, params)
        K_T = self.isothermal_bulk_modulus(P, T, V, params)
        dPdT = gr*C_V/V
        C_Ve = self._elec_molar_heat_capacity_v(T, x, params)
        dPdT_e = params['g']*C_Ve/V
        alpha = (dPdT+dPdT_e)/K_T
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
        K_T = self.isothermal_bulk_modulus(P, T, V, params)
        alpha = self.thermal_expansivity(P, T, V, params)
        C_V = self.molar_heat_capacity_v(P, T, V, params)
        K_S = K_T+V*T*(alpha*K_T)**2/C_V
        return K_S

    def pressure(self, T, V, params):
        """
        Returns pressure [Pa] as a function of temperature [K] and volume[m^3]
        (EQ 2, 10, 17).
        """
        T_0 = params['T_0']
        x = V/params['V_0']
        P = self._vinet_pressure(x, params)+\
            self._lennard_jones_pressure(V, params)+\
            self._einstein_thermal_pressure(T, V, params)-\
            self._einstein_thermal_pressure(T_0, V, params)+\
            self._elec_pressure(T, V, params)-\
            self._elec_pressure(T_0, V, params)
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
        x = V/params['V_0']
        theta = self._einstein_temperature(x, params)
        S = self._einstein_entropy(T, theta, params)+\
            self._elec_entropy(T, x, params)+\
            self._mag_entropy(T, params)+\
            self._liq_entropy(params)
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
        T_0 = params['T_0']
        x = V/params['V_0']
        theta = self._einstein_temperature(x, params)
        F = params['U_0']+\
            self._vinet_molar_internal_energy(P, T, V, params)+\
            self._lennard_jones_internal_energy(V, params)+\
            self._helmholtz_free_energy(T, theta, params)-\
            self._helmholtz_free_energy(T_0, theta, params)+\
            self._elec_helmholtz_free_energy(T, x, params)-\
            self._elec_helmholtz_free_energy(T_0, x, params)+\
            self._mag_helmholtz_free_energy(T, params)-\
            self._mag_helmholtz_free_energy(T_0, params)+\
            self._liq_helmoltz_free_energy(T, params)-\
            self._liq_helmoltz_free_energy(T_0, params)
        return F

###############################################################################

    def _einstein_temperature(self, x, params):
        """
        Computes the Einstein temperature from the parameter x = V/V_0
        (molar volumes) (EQ 15).
        """
        return params['T_einstein_0']*x**(-params['grueneisen_inf'])*\
               np.exp((params['grueneisen_0']-params['grueneisen_inf'])/\
               params['beta']*(1-x**params['beta']))

    def _grueneisen_parameter(self, x, params):
        """
        Computes the grueneisen parameter according to Altshuler form, from the
        parameter x = V/V_0 (molar volumes) (EQ 13).
        """
        gr = params['grueneisen_inf']+(params['grueneisen_0']-\
        params['grueneisen_inf'])*x**params['beta']
        return gr

    def _q_parameter(self, x, gr, params):
        """
        Computes the q parameter according to Altshuler form, from the
        parameter x = V/V_0 and the grueneisen parameter (EQ 14).
        """
        return params['beta']*x**params['beta']*(params['grueneisen_0']-\
        params['grueneisen_inf'])/gr

    def _einstein_thermal_pressure(self, T, V, params):
        """
        Computes the thermal pressure from Einstein model (EQ 10).
        """
        x = V/params['V_0']
        theta = self._einstein_temperature(x, params)
        gr = self._grueneisen_parameter(x, params)
        P_th = gr/V*self._einstein_thermal_energy(T, theta, params)
        return P_th

    def _elec_pressure(self, T, V, params):
        """
        Computes the electronic contribution to the pressure (EQ 17).
        """
        x = V/params['V_0']
        return params['g']/V*self._elec_energy(T, x, params)

    def _elec_energy(self, T, x, params):
        """
        Computes the electronic contribution to the internal energy (EQ 17).
        """
        nR = params['n']*constants.gas_constant
        return 3/2*nR*params['e_0']*x**params['g']*T**2

    def _helmholtz_free_energy(self, T, theta, params):
        """
        Computes Helmholtz free energy from Einstein model (high temperature
        limit of the Debye model) (EQ 6).
        """
        x = theta/T
        nR = params['n']*constants.gas_constant
        F = 3*nR*T*np.log(1.0-np.exp(-x))
        return F

    def _einstein_molar_heat_capacity_v(self, T, V, params):
        """
        Computes the molar heat capacity at constant volume from Einstein model
        (EQ 9).
        """
        theta = self._einstein_temperature(V/params['V_0'], params)
        x = theta/T
        nR = params['n']*constants.gas_constant
        C_V = 3.0*nR*(x*x*np.exp(x)/np.power(np.exp(x)-1.0, 2.0))
        return C_V

    def _einstein_entropy(self, T, theta, params):
        """
        Computes the entropy from Einstein model (EQ 7).
        """
        x = theta/T
        nR = params['n']*constants.gas_constant
        S = 3*nR*(-np.log(1.0-np.exp(-x))+x/(np.exp(x)-1.0))
        return S

    def _elec_entropy(self, T, x, params):
        """
        Computes the electronic entropy (EQ 17).
        """
        nR = params['n']*constants.gas_constant
        return 3*nR*params['e_0']*x**params['g']*T

    def _elec_molar_heat_capacity_v(self, T, x, params):
        """
        Computes electronic contribution to the molar heat capacity at constant
        volume (EQ 17).
        """
        nR = params['n']*constants.gas_constant
        return 3*nR*params['e_0']*x**params['g']*T

    def _einstein_thermal_bulk_modulus(self, T, V, params):
        """
        Computes the thermal correction to the isothermal Bulk modulus from
        Einstein model (EQ 11).
        """
        x = V/params['V_0']
        gr = self._grueneisen_parameter(x, params)
        q = self._q_parameter(x, gr, params)
        K_th = self._einstein_thermal_pressure(T, V, params)*(1+gr-q)-\
               gr**2*T*self._einstein_molar_heat_capacity_v(T, V, params)/V
        return K_th

    def _elec_helmholtz_free_energy(self, T, x, params):
        """
        Computes the electronic contribution to the Helmholtz free
        energy (EQ 17).
        """
        nR = params['n']*constants.gas_constant
        return -3./2*nR*params['e_0']*x**params['g']*T**2

    def _elec_thermal_bulk_modulus(self, T, V, params):
        """
        Computes the electronic contribution to the thermal Bulk modulus
        (EQ 17).
        """
        return self._elec_pressure(T, V, params)*(1.0-params['g'])

    def _liq_helmoltz_free_energy(self, T, params):
        """
        Computes the liquid contribution to the Helmholtz free energy (EQ 22).
        """
        nR = params['n']*constants.gas_constant
        F_liq = -params['a_s']*nR*T
        return F_liq

    def _liq_entropy(self, params):
        """
        Computes the liquid contribution to the entropy from F_liq (EQ 2).
        """
        nR = params['n']*constants.gas_constant
        S_liq = params['a_s']*nR
        return S_liq

    def _mag_helmholtz_free_energy(self, T, params):
        """
        Computes the magnetic contribution to the entropy from F_mag (EQ 19).
        """
        if params['Tc'] > 0:
            p = 0.4
            f = self._mag_function(p,T/params['Tc'])
            nR = params['n']*constants.gas_constant
            F_mag = nR*T*np.log(params['B_0']+1)*(f-1)
        else:
            F_mag = 0.
        return F_mag

    def _mag_entropy(self, T, params):
        """
        Computes the magnetic contribution to the entropy from F_mag (EQ 19).
        """
        if params['Tc'] > 0:
            p = 0.4
            f = self._mag_function(p,T/params['Tc'])
            fp = self._mag_function_prime(p,T/params['Tc'])
            nR = params['n']*constants.gas_constant
            S_mag = -nR*np.log(params['B_0']+1)*(f-1+T*fp/params['Tc'])
        else:
            S_mag = 0.
        return S_mag

    def _mag_molar_heat_capacity_v(self, T, params):
        """
        Computes the magnetic contribution to the molar heat capacity at
        constant volume, from F_mag (EQ 19).
        """
        if params['Tc'] > 0:
            p = 0.4
            f = self._mag_function(p,T/params['Tc'])
            fp = self._mag_function_prime(p,T/params['Tc'])
            fpp = self._mag_function_primeprime(p,T/params['Tc'])
            nR = params['n']*constants.gas_constant
            C_Vmag = -nR*T/params['Tc']*np.log(params['B_0']+1)*\
                     (2*fp+T/params['Tc']*fpp)
        else:
            C_Vmag = 0.
        return C_Vmag

    def _mag_function(self,p,tau):
        """
        Computes the magnetic function f (EQ 20, 21).
        """
        D = 518/1125+11692/15975*(1/p-1)
        if tau <= 1:
            f = 1-(79/tau/140/p+474/497*(1/p-1)*\
                (tau**3/6+tau**9/135+tau**15/600))/D
        else:
            f = -(tau**(-5)/10+tau**(-15)/315+tau**(-25)/1500)/D
        return f

    def _mag_function_prime(self,p,tau):
        """
        Computes the magnetic function f derivative (EQ 20, 21).
        """
        D = 518/1125+11692/15975*(1/p-1)
        if tau <= 1:
            fp = (79/140/tau**2/p-474/497*(1/p-1)*\
                 (3*tau**2/6+9*tau**8/135+15*tau**14/600))/D
        else:
            fp = (5*tau**(-6)/10+15*tau**(-16)/315+25*tau**(-26)/1500)/D
        return fp

    def _mag_function_primeprime(self,p,tau):
        """
        Computes the magnetic function f second derivative (EQ 20, 21).
        """
        D = 518/1125+11692/15975*(1/p-1)
        if tau <= 1:
            fpp = -(2*79/tau**3/140/p+474/497*(1/p-1)*\
                  (2*3*tau/6+8*9*tau**7/135+14*15*tau**13/600))/D
        else:
            fpp = -(6*5*tau**(-7)/10+16*15*\
                  tau**(-17)/315+26*25*tau**(-27)/1500)/D
        return fpp

    def _einstein_thermal_energy(self, T, theta, params):
        """
        Computes internal energy from Einstein model (high temperature
        limit of the Debye model) (EQ 8).
        """
        x = theta/T
        nR = params['n']*constants.gas_constant
        E_th = 3.*nR*theta*(1./(np.exp(x)-1.0))
        return E_th

    def _vinet_pressure(self, x, params):
        """
        Computes the pressure from the cold Vinet EOS (EQ 2).
        """
        eta = (3./2.)*(params['Kprime_0']-1.)
        return 3.*params['K_0']*(pow(x, -2./3.))*(1.-(pow(x, 1./3.))) \
               *np.exp(eta*(1.-pow(x, 1./3.)))

    def _vinet_bulk_modulus(self, V, params):
        """
        Computes the Bulk modulus from the Vinet EOS (EQ 3).
        """
        x = V/params['V_0']
        eta = (3./2.)*(params['Kprime_0']-1.)
        K = (params['K_0']*pow(x, -2./3.))*\
            (1+((eta*pow(x, 1./3.)+1.)*(1.-pow(x, 1./3.))))*\
            np.exp(eta*(1.-pow(x, 1./3.)))
        return K

    def _vinet_molar_internal_energy(self, P, T, V, params):
        """
        Computes the Vinet EOS contribution to the internal energy (EQ 5).
        """
        X = pow(V/params['V_0'], 1./3.)
        eta = (3./2.)*(params['Kprime_0']-1.)
        intPdV = (9.* params['V_0']*params['K_0']/(eta*eta)*\
                 ((1.-eta*(1.-X))*np.exp(eta*(1.-X))-1.))
        return -intPdV

    def _lennard_jones_internal_energy(self, V, params):
        """
        Computes the Lennard-Jones contribution to energy for large molar volumes
        """
        if V > params['V_lj']:
            rho = params['molar_mass']/V
            f1 = params['lj_f1']
            f2 = params['lj_f2']
            f3 = params['lj_f3']
            flj = params['lj_flj']
            Ecoh = params['lj_Ecoh']
            return f1*rho**f2 - f3*rho**flj + Ecoh
        else: return 0.

    def _lennard_jones_pressure(self, V, params):
        """
        Computes the Lennard-Jones contribution to pressure for large molar volumes.
        P = -dE/dV|s
        """
        if V > params['V_lj']:
            rho = params['molar_mass']/V
            drhodV = -params['molar_mass']/V**2
            f1 = params['lj_f1']
            f2 = params['lj_f2']
            f3 = params['lj_f3']
            flj = params['lj_flj']
            Ecoh = params['lj_Ecoh']
            return - (f1*f2*drhodV*rho**(f2-1.) - f3*flj*drhodV*rho**(flj-1.))
        else: return 0.
    
    def _lennard_jones_bulk_modulus(self, V, params):
        """
        Computes the Lennard-Jones contribution to pressure for large molar volumes.
        K = -VdP/dV|s
        """
        if V > params['V_lj']:
            rho = params['molar_mass']/V
            drhodV = -params['molar_mass']/V**2
            ddrhodV = 2*params['molar_mass']/V**3
            f1 = params['lj_f1']
            f2 = params['lj_f2']
            f3 = params['lj_f3']
            flj = params['lj_flj']
            Ecoh = params['lj_Ecoh']
            return V*(f1*f2*ddrhodV*rho**(f2-1.)   + f1*f2*drhodV*(f2-1.)*drhodV*rho**(f2-2.) -\
                      f3*flj*ddrhodV*rho**(flj-1.) - f3*flj*drhodV*(flj-1.)*drhodV*rho**(f2-2.))
        else: return 0.
    
    def validate_parameters(self, params):
        """
        Check for existence and validity of the parameters
        """
        if 'T_0' not in params:  params['T_0']  = 300.
        if 'U_0' not in params:  params['U_0']  = 0.
        if 'Tc' not in params:   params['Tc']   = 0.
        if 'B_0' not in params:  params['B_0']  = 0.
        if 'a_s' not in params:  params['a_s']  = 0.
        if 'V_lj' not in params: params['V_lj'] = float('inf')
            
        # Initialize limits
        if 'P_min' not in params: params['P_min'] = -float('inf')
        if 'P_max' not in params: params['P_max'] = float('inf')
        if 'V_min' not in params: params['V_min'] = -float('inf')
        if 'V_max' not in params: params['V_max'] = float('inf')
        if 'T_min' not in params: params['T_min'] = -float('inf')
        if 'T_max' not in params: params['T_max'] = float('inf')

        # Now check all the required keys for the
        # thermal part of the EoS are in the dictionary
        expected_keys = ['V_0', 'K_0', 'Kprime_0', 'T_einstein_0', \
                         'grueneisen_0', 'beta', 'grueneisen_inf', \
                         'e_0', 'g', 'molar_mass', 'n']
        for k in expected_keys:
            if k not in params:
                raise KeyError('params object missing parameter : '+k)

        # Finally, check that the values are reasonable.
        if params['T_0'] < 0.:
            warnings.warn('Unusual value for T_0', stacklevel = 2)
        if params['molar_mass'] < 0.001 or params['molar_mass'] > 1.:
            warnings.warn('Unusual value for molar_mass', stacklevel = 2)
        if params['n'] < 1. or params['n'] > 1000.:
            warnings.warn('Unusual value for n', stacklevel = 2)
