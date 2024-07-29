# Evolution models for classic feedback bubble solutions
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as aconsts
import quantities

class Bubble():
    def __init__(self, rho0):
        self.rho0 = rho0
        t1 = u.get_physical_type(rho0)=="mass density"
        if not(t1):
            print("Units of rho0 are off")
            assert(False)

    def radius(self, t):
        return 0.0
    
    def velocity(self, t):
        return 0.0
    
    def momentum(self, t):
        return 0.0

    def pressure(self, t):
        return 0.0

class Spitzer(Bubble):
    # Spitzer solution for a photo-ionized gas bubble
    # includes the Hosokawa & Inutsuka (2006) correction
    def __init__(self, rho0, Q0, ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s), muH = 1.4):
        super().__init__(rho0)
        self.Q0 = Q0
        self.ci = ci
        self.alphaB = alphaB
        self.muH = muH
        t1 = u.get_physical_type(ci)=="speed"
        t2 = u.get_physical_type(Q0)=="frequency"
        t3 = u.get_physical_type(alphaB)=="volumetric flow rate"
        t4 = u.get_physical_type(muH)=="dimensionless"
        if not(t1):
            print("Units of ci are off")
            assert(False)
        if not(t2):
            print("Units of Q0 are incorrect")
            assert(False)
        if not(t3):
            print("Units of alphaB are off")
            assert(False)
        if not(t4):
            print("Units of muH are off")
            assert(False)

        self.nbar = rho0/(muH*aconsts.m_p)
        self.RSt = quantities.RSt(Q0, self.nbar, alphaB=alphaB)
        self.tdio = quantities.Tdion(self.Q0, self.nbar, ci=ci, alphaB=alphaB)

    def rhoi(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        rhoi_sp = self.rho0*(1 + 7*t/(4*self.tdio))**(-3./2)
        return rhoi_sp.to("solMass/pc3")

    def radius(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        r_sp = self.Rst*(1 + 7*t/(4*self.tdio))**(4./7)
        return r_sp.to("pc")
    
    def velocity(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        v_sp = (self.RSt/self.tdio)*(1 + 7*t/(4*self.tdio))**(-3./7)
        return v_sp.to("km/s")
    
    def momentum(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        prefac = 4*np.pi*self.rho0*self.RSt**4/(3*self.tdio)
        pr_sp = prefac*(1 + 7*t/(4*self.tdio))**(9./7)*(1 - (self.RSt/self.radius(t))**1.5)
        return pr_sp.to("solMass*km/s")
    
    def pressure(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        press_sp = self.rhoi(t)*self.ci**2

        return (press_sp/aconsts.k_B).to("K/cm3")


class Weaver(Bubble):
    # Weaver solution for a wind bubble
    def __init__(self, rho0, Lwind):
        super().__init__(rho0)
        self.Lwind = Lwind
        t1 = u.get_physical_type(Lwind)=="power"
        if not(t1):
            print("Units of Lwind are incorrect")
            assert(False)

    def radius(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        r_we = (125*self.Lwind*(t**3)/(154*np.pi*self.rho0))**(1./5)
        return r_we.to("pc")
    
    def velocity(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        v_we = 0.6*self.radius(t)/t
        return v_we.to("km/s")
    
    def momentum(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        pr_we = 4*np.pi*self.rho0*self.radius(t)**3*self.velocity(t)/3
        return pr_we.to("solMass*km/s")
    
    def pressure(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        press_we = (10./33)*self.Lwind*self.t/((4*np.pi/3)*self.radius(t)**3)
        return (press_we/aconsts.k_B).to("K/cm3")
    
class MomentumDriven(Bubble):
    # Momentum-driven bubble solution
    def __init__(self, rho0, pdotw):
        super().__init__(rho0)
        self.pdotw = pdotw
        t1 = u.get_physical_type(pdotw)=="force"
        if not(t1):
            print("Units of pdotw are incorrect")
            assert(False)

    def radius(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        r_md = ((3*self.pdotw*t)/(2*np.pi*self.rho0))**(1./2)
        return r_md.to("pc")
    
    def velocity(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        v_md = 0.5*self.radius(t)/t
        return v_md.to("km/s")
    
    def momentum(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        pr_md = self.pdotw*t
        return pr_md.to("solMass*km/s")
    
    def pressure(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        press_md = self.pdotw/(4*np.pi*self.radius(t)**2)
        return (press_md/aconsts.k_B).to("K/cm3")