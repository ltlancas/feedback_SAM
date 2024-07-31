# Models to the evolution of feedback bubbles
# stellar feedback from massive stars
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as aconsts
import quantities
import classic_bubbles as cb
from classic_bubbles import Bubble
from scipy.integrate import solve_ivp


class JointBubbleUncoupled(Bubble):
    # Joint solution for the evolution of a photo-ionized gas bubble
    # and a wind bubble in force balance with each other
    # assumes that the bubbles are uncoupled and evolve independently
    # up until t_eq, the equilibration time
    def __init__(self, rho0, Q0, pdotw,
                 ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s),
                 muH = 1.4, ic1op = 0, ic2op = 4):
        super().__init__(rho0)
        # set parameters
        self.Q0 = Q0
        self.pdotw = pdotw
        self.ci = ci
        self.alphaB = alphaB
        self.muH = muH
        self.ic1op = ic1op
        self.ic2op = ic2op
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
        self.teq = quantities.Teq(pdotw, rho0, ci=ci)
        self.Req = quantities.Req(pdotw, rho0, ci=ci)
        self.Rch = quantities.Rch(Q0, self.nbar, pdotw, rho0, ci=ci, alphaB=alphaB)
        self.tdio = quantities.Tdion(Q0, self.nbar, ci=ci, alphaB=alphaB)

        self.eta = self.RSt/self.Rch

        # Separate Spitzer solution
        self.spitz_bubble = cb.Spitzer(rho0, Q0, ci=ci, alphaB=alphaB, muH=muH)
        # separate momentum-driven wind bubble
        self.wind_bubble = cb.MomentumDriven(rho0, pdotw)
        # call ODE integrator to get the joint evolution solution
        (self.chi, self.xiw, self.xii, self.psii) = self.joint_evol()

    def joint_evol(self):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble
        # eta : the RSt/Rch ratio, free parameter of the model
        # ic1op : the choice of initial condition for Rw/Rch
        # ic2op : the choice of initial condition for dR_i/dt

        eta = self.eta.value

        # set the initial conditions
        if (self.ic1op==0):
            xiw0 = eta**0.75
        elif (self.ic1op==1):
            xiw0 = eta**1.5
        else:
            print("Bad option for initial condition 1")
            assert(False)
        
        xii0 = xiw0*((1+xiw0)**(1./3))

        # initial condition for derivative, psi
        if (self.ic2op==0):
            psi0 = 0
        elif (self.ic2op==1):
            psi0 = eta
        elif (self.ic2op==2):
            #psi0 = np.pi*(eta**3.25)/(np.sqrt(2)*(xii0**3))
            psi0 = 3*(eta**3.25)/(2*np.sqrt(2)*(xii0**3))
        elif (self.ic2op==3):
            RiRst = 1 + 7*np.sqrt(2)*np.pi*(eta**-0.25)/18
            psi0 = eta*(RiRst**(3./7))*(1 - RiRst**(-6./7))
        elif (self.ic2op==4):
            RiRst = 1 + 7*np.sqrt(2)*np.pi*(eta**-0.25)/18
            psi0 = eta*(RiRst**(3./7))*(1 - RiRst**(-6./7))
            #psi0 += np.pi*(eta**3.25)/(np.sqrt(2)*(xii0**3))
            psi0 += 3*(eta**3.25)/(2*np.sqrt(2)*(xii0**3))
        else:
            print("Bad option for initial condition 2")
            assert(False)

        # pre-calculate the relationship between xii and xiw
        xii_range = np.linspace(xii0/2,100*xii0,1000)
        xiw_prec = []
        for xi in xii_range:
            p = [1.0, 1.0, 0., 0., -1*(xi**3)]
            xiw_prec.append(np.real(np.roots(p)[-1]))

        xiw_prec = np.array(xiw_prec)

        # defin the differential equations
        def derivs(chi,y):
            xii = y[0]
            psi = y[1]
            xiw = np.interp(xii,xii_range,xiw_prec)
            t1 = (2.25*(eta**3.5)*(1 + xiw)**(2./3))/(xii**3)
            t2 = 3*(psi**2)/xii
            return (psi,t1-t2)

        # use solve_ivp to get solution
        sol = solve_ivp(derivs,[0,100],[xii0,psi0])

        chi = sol["t"]
        xiw = np.interp(sol["y"][0],xii_range,xiw_prec)
        xii = sol["y"][0]
        psii = sol["y"][1]

        return(chi,xiw,xii,psii)


    def radius(self, t):
        # Returns the radius of the ionized bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        ri = self.spitz_bubble.radius(t)*(t<self.teq)
        ri += np.interp(t-self.teq,self.chi*self.tdio,self.xii*self.Rch)*(t>self.teq)
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
        rw += np.interp(t-self.teq,self.chi*self.tdio,self.xiw*self.Rch)*(t>self.teq)
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
        vi += np.interp(t-self.teq,self.chi*self.tdio,self.psii*self.Rch/self.tdio)*(t>self.teq)
        return vi.to("km/s")
    
    def momentum(self, t):
        # returns the momentum carried by the joint bubble at time t
        # t : the time
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            print("Units of t are off")
            assert(False)
        prefac = 4*np.pi*self.rho0*self.Rch**4/(3*self.tdio)
        pr = prefac*np.interp(t-self.teq,self.chi*self.tdio, self.psii*self.xii**3)*(t>self.teq)
        pr += self.spitz_bubble.momentum(t)*(t<self.teq)
        pr += self.wind_bubble.momentum(t)*(t<self.teq)
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
        