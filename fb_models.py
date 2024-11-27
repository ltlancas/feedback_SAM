# Models to the evolution of feedback bubbles
# stellar feedback from massive stars
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as aconsts
import quantities
from scipy.integrate import solve_ivp

#########################################################################################
########################### CLASSICAL BUBBLE EVOLUTION MODELS ###########################
#########################################################################################

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
        r_sp = self.RSt*(1 + 7*t/(4*self.tdio))**(4./7)
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
        press_we = (10./33)*self.Lwind*t/((4*np.pi/3)*self.radius(t)**3)
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
        r_md = ((3*self.pdotw*t**2)/(2*np.pi*self.rho0))**(1./4)
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


#########################################################################################
################################# JOINT EVOLUTION MODELS ################################
#########################################################################################

class JointBubbleMDUncoupled(Bubble):
    # Joint solution for the evolution of a photo-ionized gas bubble
    # and a wind bubble in force balance with each other
    # assumes that the bubbles are uncoupled and evolve independently
    # up until t_eq, the equilibration time
    def __init__(self, rho0, Q0, pdotw,
                 ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s), muH = 1.4):
        super().__init__(rho0)
        # set parameters
        self.Q0 = Q0
        self.pdotw = pdotw
        self.ci = ci
        self.alphaB = alphaB
        self.muH = muH
        # check that the units are correct
        t1 = u.get_physical_type(ci)=="speed"
        t2 = u.get_physical_type(Q0)=="frequency"
        t3 = u.get_physical_type(pdotw)=="force"
        t4 = u.get_physical_type(alphaB)=="volumetric flow rate"
        t5 = u.get_physical_type(muH)=="dimensionless"
        if not(t1):
            print("Units of ci are off")
            assert(False)
        if not(t2):
            print("Units of Q0 are incorrect")
            assert(False)
        if not(t3):
            print("Units of pdotw are incorrect")
            assert(False)
        if not(t4):
            print("Units of alphaB are off")
            assert(False)
        if not(t5):
            print("Units of muH are off")
            assert(False)

        self.nbar = rho0/(muH*aconsts.m_p)
        self.RSt = quantities.RSt(Q0, self.nbar, alphaB=alphaB)
        self.teq = quantities.Teq_MD(pdotw, rho0, ci=ci)
        self.Req = quantities.Req_MD(pdotw, rho0, ci=ci)
        self.Rch = quantities.Rch(Q0, self.nbar, pdotw, rho0, ci=ci, alphaB=alphaB)
        self.tdio = quantities.Tdion(Q0, self.nbar, ci=ci, alphaB=alphaB)
        self.tff = quantities.Tff(rho0)
        self.pscl = ((4*np.pi/3)*rho0*(self.Req**4)/self.tdio).to("solMass*km/s")

        self.eta = (self.Req/self.RSt).to(" ").value

        # Separate Spitzer solution
        self.spitz_bubble = Spitzer(rho0, Q0, ci=ci, alphaB=alphaB, muH=muH)
        # separate momentum-driven wind bubble
        self.wind_bubble = MomentumDriven(rho0, pdotw)
        # call ODE integrator to get the joint evolution solution
        self.joint_sol = self.joint_evol()

    def get_xiw(self, xii):
        xiw = []
        for xi in xii:
            p = [self.eta**-3, 1.0, 0., 0., -1*(xi**3)]
            xiw.append(np.real(np.roots(p)[-1]))
        return np.array(xiw)

    def joint_evol(self):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble
        # eta : the Req/RSt ratio, free parameter of the model

        eta = self.eta

        xiw0 = 1
        xii0 = ((1+(eta**-3))**(1./3))
        momentum_tot = self.wind_bubble.momentum(self.teq) + self.spitz_bubble.momentum(self.teq)
        mass_tot = 4*np.pi*self.rho0*((self.Req*xii0)**3)/3
        psi0 = (((momentum_tot/mass_tot)*(self.tdio/self.Req)).to(" ")).value

        # pre-calculate the relationship between xii and xiw
        xii_range = np.linspace(xii0/2,100*xii0,1000)
        xiw_prec = self.get_xiw(xii_range)

        # defin the differential equations
        def derivs(chi,y):
            xii = y[0]
            psi = y[1]
            xiw = np.interp(xii,xii_range,xiw_prec)
            t1 = (2.25*(eta**-2)*(1 + (eta**-3)*xiw)**(2./3))/(xii**3)
            t2 = 3*(psi**2)/xii
            return (psi,t1-t2)

        # use solve_ivp to get solution
        return solve_ivp(derivs,[0,100],[xii0,psi0],dense_output=True)


    def radius(self, t):
        # Returns the radius of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        ri = self.spitz_bubble.radius(t)*(t<self.teq)
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        ri += solution[0]*self.Req*(t>self.teq)
        return ri.to("pc")

    def wind_radius(self, t):
        # Returns the radius of the wind bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        # up until teq the wind bubble follows the normal momentum-driven solution
        rw = self.wind_bubble.radius(t)*(t<self.teq)
        # afterwards it follows the joint evolution solution
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        xiw = self.get_xiw(solution[0])
        rw += xiw*self.Req*(t>self.teq)
        return rw.to("pc")

    def velocity(self, t):
        # Returns the velocity of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        # up until teq the ionized bubble follows the Spitzer solution
        vi = self.spitz_bubble.velocity(t)*(t<self.teq)
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        vi += solution[1]*self.Req/self.tdio*(t>self.teq)
        return vi.to("km/s")
    
    def momentum(self, t):
        # returns the momentum carried by the joint bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        prefac = self.pscl
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        pr = prefac*solution[1]*solution[0]**3*(t>self.teq)
        pr += self.spitz_bubble.momentum(t)*(t<self.teq)
        pr += self.wind_bubble.momentum(t)*(t<self.teq)
        return pr.to("solMass*km/s")

    def momentum_uncoupled(self, t):
        # returns the momentum carried by the joint bubble at time t
        # if the two constituent bubbles evolved independently
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        pr = self.spitz_bubble.momentum(t)
        pr += self.wind_bubble.momentum(t)
        return pr.to("solMass*km/s")
    
    def pressure(self, t):
        # returns the pressure of the wind bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        press = self.pdotw/(4*np.pi*self.wind_radius(t)**2)
        return (press/aconsts.k_B).to("K/cm3")
    
    def pressure_ionized(self, t):
        # returns the pressure of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        press = self.spitz_bubble.pressure(t)*(t<self.teq)
        press += self.pressure(t)*(t>self.teq)
        return press

class JointBubbleEDUncoupled(Bubble):
    # Joint solution for the evolution of a photo-ionized gas bubble
    # and a wind bubble in force balance with each other
    # assumes that the bubbles are uncoupled and evolve independently
    # up until t_eq, the equilibration time
    def __init__(self, rho0, Q0, Lwind,
                 ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s), muH = 1.4):
        super().__init__(rho0)
        # set parameters
        self.Q0 = Q0
        self.Lwind = Lwind
        self.ci = ci
        self.alphaB = alphaB
        self.muH = muH
        # check that the units are correct
        t1 = u.get_physical_type(ci)=="speed"
        t2 = u.get_physical_type(Q0)=="frequency"
        t3 = u.get_physical_type(Lwind)=="power"
        t4 = u.get_physical_type(alphaB)=="volumetric flow rate"
        t5 = u.get_physical_type(muH)=="dimensionless"
        if not(t1):
            print("Units of ci are off")
            assert(False)
        if not(t2):
            print("Units of Q0 are incorrect")
            assert(False)
        if not(t3):
            print("Units of pdotw are incorrect")
            assert(False)
        if not(t4):
            print("Units of alphaB are off")
            assert(False)
        if not(t5):
            print("Units of muH are off")
            assert(False)

        self.nbar = rho0/(muH*aconsts.m_p)
        self.RSt = quantities.RSt(Q0, self.nbar, alphaB=alphaB)
        self.teq = quantities.Teq_ED(Lwind, rho0, ci=ci)
        self.Req = quantities.Req_ED(Lwind, rho0, ci=ci)
        self.tdio = quantities.Tdion(Q0, self.nbar, ci=ci, alphaB=alphaB)
        self.tff = quantities.Tff(rho0)
        self.pscl = ((4*np.pi/3)*rho0*(self.Req**4)/self.tdio).to("solMass*km/s")

        self.eta = (self.Req/self.RSt).to(" ").value

        # Separate Spitzer solution
        self.spitz_bubble = Spitzer(rho0, Q0, ci=ci, alphaB=alphaB, muH=muH)
        # separate momentum-driven wind bubble
        self.wind_bubble = Weaver(rho0, Lwind)
        # call ODE integrator to get the joint evolution solution
        self.joint_sol = self.joint_evol()

    def joint_evol(self):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble
        # eta : the Req/RSt ratio, free parameter of the model

        eta = self.eta

        xiw0 = 1
        xii0 = ((1+(eta**-3))**(1./3))
        Et0 = (2./11)*np.sqrt(7./3)*eta
        momentum_tot = self.wind_bubble.momentum(self.teq) + self.spitz_bubble.momentum(self.teq)
        mass_tot = 4*np.pi*self.rho0*((self.Req*xii0)**3)/3
        Mi0 = (((momentum_tot/mass_tot)/self.ci).to(" ")).value


        # defin the differential equations
        def derivs(chi,y):
            (xii,Mi,xiw,Et) = y
            Pt = (11./2)*np.sqrt(3/7)*Et*(xiw**-3)/eta
            A = 2/(3*((xiw*eta)**3)*(Pt**2))
            dlnxii_dchi = (2/np.sqrt(3))*Mi/xii

            dxii_dchi = dlnxii_dchi*xii
            dMi_dchi = (3*np.sqrt(3)/(2*eta*xii))*(Pt - Mi**2)
            dlnxiw_dchi = (((xii/xiw)**3)*dlnxii_dchi + A/Et)/(1 + 5*A)
            dxiw_dchi = dlnxiw_dchi*xiw
            dlnEt_chi = 1./Et - 2*dlnxiw_dchi
            dEt_dchi = dlnEt_chi*Et
            return (dxii_dchi,dMi_dchi,dxiw_dchi,dEt_dchi)

        # use solve_ivp to get solution
        return solve_ivp(derivs,[0,100],[xii0,Mi0,xiw0,Et0],dense_output=True)


    def radius(self, t):
        # Returns the radius of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        ri = self.spitz_bubble.radius(t)*(t<self.teq)
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        ri += solution[0]*self.Req*(t>self.teq)
        return ri.to("pc")

    def wind_radius(self, t):
        # Returns the radius of the wind bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        # up until teq the wind bubble follows the normal momentum-driven solution
        rw = self.wind_bubble.radius(t)*(t<self.teq)
        # afterwards it follows the joint evolution solution
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        xiw = solution[2]
        rw += xiw*self.Req*(t>self.teq)
        return rw.to("pc")

    def velocity(self, t):
        # Returns the velocity of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        # up until teq the ionized bubble follows the Spitzer solution
        vi = self.spitz_bubble.velocity(t)*(t<self.teq)
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        vi += solution[1]*self.ci*(t>self.teq)
        return vi.to("km/s")
    
    def momentum(self, t):
        # returns the momentum carried by the joint bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        prefac = (4*np.pi/3)*self.Req**3*self.rho0*self.ci
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        pr = prefac*solution[1]*solution[0]**3*(t>self.teq)
        pr += self.spitz_bubble.momentum(t)*(t<self.teq)
        pr += self.wind_bubble.momentum(t)*(t<self.teq)
        return pr.to("solMass*km/s")

    def momentum_uncoupled(self, t):
        # returns the momentum carried by the joint bubble at time t
        # if the two constituent bubbles evolved independently
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        pr = self.spitz_bubble.momentum(t)
        pr += self.wind_bubble.momentum(t)
        return pr.to("solMass*km/s")
    
    def pressure(self, t):
        # returns the pressure of the wind bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        press = self.wind_bubble.pressure(t)*(t<self.teq)
        chi = ((t-self.teq)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        Pt = (11./2)*np.sqrt(3/7)*solution[3]*(solution[2]**-3)/self.eta
        Pt = Pt*self.rho0*self.ci**2
        press += Pt*(t>self.teq)/aconsts.k_B
        return (press).to("K/cm3")
    
    def pressure_ionized(self, t):
        # returns the pressure of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        press = self.spitz_bubble.pressure(t)*(t<self.teq)
        press += self.pressure(t)*(t>self.teq)
        return press


class JointBubbleCoupled(Bubble):
    # Joint solution for the evolution of a photo-ionized gas bubble
    # that includes an "early" phase where the bubbles are not in force balance
    # with one another
    def __init__(self, rho0, Q0, Mdotw, Vwind,
                 ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s),
                 muH = 1.4, alphap = 1, dynamic_density=False):
        # dynamic_density : whether to track the ionized gas denisty as a 
        #    dynamical variable with it's own ODE
        super().__init__(rho0)
        # set parameters
        self.Q0 = Q0
        self.mdotw = Mdotw
        self.vwind = Vwind
        self.ci = ci
        self.alphaB = alphaB
        self.muH = muH
        # check that the units are correct
        t1 = u.get_physical_type(ci)=="speed"
        t2 = u.get_physical_type(Q0)=="frequency"
        t3 = u.get_physical_type(Mdotw*u.s)=="mass"
        t4 = u.get_physical_type(Vwind)=="speed"
        t5 = u.get_physical_type(alphaB)=="volumetric flow rate"
        t6 = u.get_physical_type(muH)=="dimensionless"
        if not(t1):
            print("Units of ci are off")
            assert(False)
        if not(t2):
            print("Units of Q0 are incorrect")
            assert(False)
        if not(t3):
            print("Units of Mdotw are incorrect")
            assert(False)
        if not(t4):
            print("Units of Vwind are off")
            assert(False)
        if not(t5):
            print("Units of alphaB are off")
            assert(False)
        if not(t6):
            print("Units of muH are off")
            assert(False)

        self.pdotw = alphap*Mdotw*Vwind
        self.Lwind = 0.5*self.pdotw*self.vwind
        self.nbar = rho0/(muH*aconsts.m_p)
        self.RSt = quantities.RSt(Q0, self.nbar, alphaB=alphaB)
        self.teq = quantities.Teq_MD(self.pdotw, rho0, ci=ci)
        self.Req = quantities.Req_MD(self.pdotw, rho0, ci=ci)
        self.Rch = quantities.Rch(Q0, self.nbar, self.pdotw, rho0, ci=ci, alphaB=alphaB)
        self.tdio = quantities.Tdion(Q0, self.nbar, ci=ci, alphaB=alphaB)
        self.tff = quantities.Tff(rho0)
        self.tcool = quantities.Tcool(self.nbar, self.Lwind)
        self.taurec0 = quantities.Tion(self.nbar,alphaB=alphaB)
        self.dynamic_density = dynamic_density

        self.eta = (self.RSt/self.Rch).to(" ").value


        # reference solutions
        self.spitz_bubble = Spitzer(rho0, Q0, ci=ci, alphaB=alphaB, muH=muH)
        self.weaver = Weaver(rho0, self.Lwind)
        self.mc_wind = MomentumDriven(rho0, self.pdotw)
        
        # call ODE integrator to get the joint evolution solution
        self.early_sol = self.early_evol()

        # when we transition between early evolution and joint evolution
        for event in self.early_sol.t_events:
            if (len(event) != 0):
                self.t_transition = (event[0]*self.tdio).to(u.Myr)
        
        sol_transition = self.early_sol.sol((self.t_transition/self.tdio).to(" ").value)
        # get total momentum at transition time
        if self.dynamic_density:
            (mushw, xiw, Machw, xii, Machi, di) = sol_transition
        else:
            (mushw, xiw, Machw, xii, Machi) = sol_transition
            # factors needed for calculating derivatives
            mrat = mushw*(Machw**2)/(xii**3 - xiw**3)
            x = np.sqrt(1 + 4*(self.eta**3)/(mrat*mushw*(Machw**2)))
            # ionized gas density ratio
            di = 0.5*mrat*(x - 1)
        self.xii_transition = xii
        self.xiw_transition = xiw
        # initial condition for velocity on joint evolution
        # based on keeping the same momentum
        mushi_transition =  xii**3 - di*(xii**3 - xiw**3) - mushw
        Mi0_transition = (mushw*Machw + mushi_transition*Machi)/xii**3
        #print(xiw, xii, xiw*((1+xiw)**(1./3)))

        self.joint_sol = self.joint_evol(xiw, Mi0_transition)

    def get_xiw(self, xii):
        xiw = []
        for xi in xii:
            p = [1.0, 1.0, 0., 0., -1*(xi**3)]
            xiw.append(np.real(np.roots(p)[-1]))
        return np.array(xiw)

    def joint_evol(self,xiw0,Machi0):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble
        # eta : the RSt/Rch ratio, free parameter of the model

        eta = self.eta
        # get xii0 from xiw0
        xii0 = xiw0*((1+xiw0)**(1./3))

        # pre-calculate the relationship between xii and xiw
        xii_range = np.linspace(xii0/2,100*xii0,1000)
        xiw_prec = self.get_xiw(xii_range)

        # define the differential equations
        def derivs(chi,y):
            (xii,Machi) = y
            xiw = np.interp(xii,xii_range,xiw_prec)
            mfac = np.sqrt(3)*eta/2
            t1 = (3*mfac*(eta**1.5)*(1 + xiw)**(2./3))/(xii**3)
            t2 = 3*mfac*(Machi**2)/xii
            return (mfac*Machi,t1-t2)

        # use solve_ivp to get solution
        return solve_ivp(derivs,[0,100],[xii0,Machi0],dense_output=True)

    def early_evol(self):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble at early times
        # before they are in force balance with each other
        # eta : the RSt/Rch ratio, free parameter of the model

        eta = self.eta

        Rw_init = self.weaver.radius(self.tcool)
        dotRw_init = 0.6*Rw_init/self.tcool

        xiw_init = (Rw_init/self.Rch).to(" ").value
        Machw_init = (dotRw_init/self.ci).to(" ").value
        mushw_init = xiw_init**3

        xii_init = np.cbrt(eta**3 + (xiw_init**3)*(1-Machw_init**2))
        Machi_init = 2/np.sqrt(3)
        if self.dynamic_density:
            di_init = 1.0
            y0 = [mushw_init, xiw_init, Machw_init, xii_init, Machi_init, di_init]
        else:
            y0 = [mushw_init, xiw_init, Machw_init, xii_init, Machi_init]

        # conditions on which to stop the evolution
        def wind_caught_up(chi, y):
            # wind radius reaches ionized gas radius
            # this really shouldn't happen
            return y[1] - y[3]

        def peq_reached(chi, y):
            # pressure equilibrium between wind and photo-ionized gas
            if self.dynamic_density:
                (mushw, xiw, Machw, xii, Machi, di) = y
                return di - eta**1.5/xiw**2
            else:
                (mushw, xiw, Machw, xii, Machi) = y
                mrat = mushw*(Machw**2)/(xii**3 - xiw**3)
                x = np.sqrt(1 + 4*(eta**3)/mrat/(mushw*(Machw**2)))
                # ionized gas density ratio
                di = 0.5*mrat*(x - 1)
                return di - eta**1.5/xiw**2
        
        def wind_subsonic(chi, y):
            # wind bubble's expansion becomes subsonic in the 
            # photo-ionized gas
            return y[2] - 1.0

        # make sure to terminate integration on gas depletion
        wind_caught_up.terminal = True
        peq_reached.terminal = True
        wind_subsonic.terminal = True

        # define the differential equations
        def derivs(chi,y):
            if self.dynamic_density:
                (mushw, xiw, Machw, xii, Machi, di) = y
            else:
                (mushw, xiw, Machw, xii, Machi) = y
            # factors needed for calculating derivatives
            mfac = np.sqrt(3)*eta/2
            mrat = mushw*(Machw**2)/(xii**3 - xiw**3)
            x = np.sqrt(1 + 4*(eta**3)/mrat/(mushw*(Machw**2)))
            # ionized gas density ratio
            if not(self.dynamic_density):
                di = 0.5*mrat*(x - 1)
            # derivatives
            dmushw = 3*mfac*(xiw**2)*Machw*di
            dxiw = mfac*Machw
            dMachw = (3*mfac*(eta**1.5) - Machw*dmushw)/mushw
            dMachi = 3*(mfac/xii)*(di - Machi**2)
            dxii_dyn = mfac*Machi
            if self.dynamic_density:
                # dimensionless initial recombination time
                chirec0 = (self.taurec0/self.tdio).to(" ").value
                chirec = 1./((mushw*Machw**4*di**2 + di**3*(xii**3 - xiw**3))/(chirec0*eta**3))
                # equilibrium ionization front position
                xii_eq = np.cbrt(xiw**3 + eta**3/(di**2) - mushw*Machw**2/di)
                dxii = dxii_dyn + (xii_eq - xii)/chirec
                # calculation of derivative of density in time
                dmrat_dchi = dmushw/mushw + 2*dMachw/Machw
                dmrat_dchi -= 3*(xii**2*dxii_dyn - xiw**2*dxiw)/(xii**3 - xiw**3)
                dmrat_dchi *= mrat
                dx_dchi = dmushw/mushw + 2*dMachw/Machw + dmrat_dchi/mrat
                dx_dchi *= -2*(eta**3)/(x*mushw*mrat*(Machw**2))
                ddi = 0.5*((x-1)*dmrat_dchi + mrat*dx_dchi)
                return (dmushw,dxiw,dMachw,dxii,dMachi,ddi)
            else:
                dxii = dxii_dyn
                return (dmushw,dxiw,dMachw,dxii,dMachi)

        # use solve_ivp to get solution
        return solve_ivp(derivs,[0,100],y0,events=[wind_caught_up, peq_reached, wind_subsonic], dense_output=True)

    def radius(self, t):
        # Returns the radius of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        echi = (t/self.tdio).to(" ").value
        esol =  self.early_sol.sol(echi)
        jchi = ((t-self.t_transition)/self.tdio).to(" ").value
        jsol = self.joint_sol.sol(jchi)
        ri = esol[3]*self.Rch*(t<self.t_transition)
        ri += jsol[0]*self.Rch*(t>=self.t_transition)
        return ri.to("pc")

    def wind_radius(self, t):
        # Returns the radius of the wind bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        echi = (t/self.tdio).to(" ").value
        esol = self.early_sol.sol(echi)
        jchi = ((t-self.t_transition)/self.tdio).to(" ").value
        jsol = self.joint_sol.sol(jchi)
        rw = esol[1]*self.Rch*(t<self.t_transition)
        rw += self.get_xiw(jsol[0])*self.Rch*(t>=self.t_transition)
        return rw.to("pc")

    def velocity(self, t):
        # Returns the velocity of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        echi = (t/self.tdio).to(" ").value
        esol = self.early_sol.sol(echi)
        jchi = ((t-self.t_transition)/self.tdio).to(" ").value
        jsol = self.joint_sol.sol(jchi)
        vi = esol[4]*self.ci*(t<self.t_transition)
        vi += jsol[1]*self.ci*(t>=self.t_transition)
        return vi.to("km/s")
    
    def wind_velocity(self, t):
        # Returns the velocity of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        echi = (t/self.tdio).to(" ").value
        esol = self.early_sol.sol(echi)
        vw = esol[2]*self.ci
        return vw.to("km/s")
    
    def momentum(self, t):
        # returns the momentum carried by the joint bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        # prefactor on momentum calculation
        prefac = 4*np.pi*self.rho0*self.Rch**3*self.ci/3
        chi = (t/self.tdio).to(" ").value
        solution =  self.early_sol.sol(chi)
        if self.dynamic_density:
            (mushw, xiw, Machw, xii, Machi, di) = solution
        else:
            (mushw, xiw, Machw, xii, Machi) = solution
            # factors needed for calculating derivatives
            mrat = mushw*(Machw**2)/(xii**3 - xiw**3)
            x = np.sqrt(1 + 4*(self.eta**3)/(mrat*mushw*(Machw**2)))
            # ionized gas density ratio
            di = 0.5*mrat*(x - 1)

        prw = prefac*mushw*Machw
        mushi = xii**3 - di*(xii**3 - xiw**3) - mushw
        pri = prefac*mushi*Machi
        pr = prw + pri
        return pr.to("solMass*km/s")
    
    def pressure(self, t):
        # returns the pressure of the wind bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        
        press = self.pdotw/(4*np.pi*self.wind_radius(t)**2)
        return (press/aconsts.k_B).to("K/cm3")

    def rhoi(self,t):
        # returns the density in the ionized gas bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        chi = (t/self.tdio).to(" ").value
        solution =  self.early_sol.sol(chi)
        if self.dynamic_density:
            (mushw, xiw, Machw, xii, Machi, di) = solution
            return (di*self.rho0).to("solMass/pc3")
        else:
            (mushw, xiw, Machw, xii, Machi) = solution
            # factors needed for calculating derivatives
            mrat = mushw*(Machw**2)/(xii**3 - xiw**3)
            x = np.sqrt(1 + 4*(self.eta**3)/(mrat*mushw*(Machw**2)))
            # ionized gas density ratio
            di = 0.5*self.rho0*mrat*(x - 1)
            return di.to("solMass/pc3")

    def pressure_ionized(self, t):
        # returns the pressure of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        return (self.rhoi(t)*self.ci**2/aconsts.k_B).to("K/cm3")