# File that contains the definitions of various quantities related to
# stellar feedback from massive stars
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as aconsts

def RSt(Q0, nbar, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # Returns the Strömgren Radius in parsecs
    # Q0 : the ionizing photon rate
    # nbar : the number density of hydrgen in the background
    # alphaB : the case B recombination rate

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(Q0)=="frequency"
    t2 = u.get_physical_type(nbar)=="number density"
    t3 = u.get_physical_type(alphaB)=="volumetric flow rate"
    if not(t1):
        print("Units of Q0 are incorrect")
        assert(False)
    if not(t2):
        print("Units of nbar are off")
        assert(False)
    if not(t3):
        print("Units of alphaB are off")
        assert(False)
    
    r_st = (3*Q0/(4*np.pi*nbar**2*alphaB))**(1./3)
    
    return r_st.to("pc")

def Req(pdotw, rhobar, ci = 10*u.km/u.s):
    # Returns the equilbration radius
    # pdotw : wind momentum input rate
    # rhobar : background mass density
    # ci : ionized gas sound speed

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(pdotw)=="force"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        print("Units of pdotw are incorrect")
        assert(False)
    if not(t2):
        print("Units of rhobar are off")
        assert(False)
    if not(t3):
        print("Units of ci are off")
        assert(False)

    r_eq = (pdotw/(4*np.pi*rhobar*ci**2))**(1./2)

    return r_eq.to("pc")

def Rch(Q0, nbar, pdotw, rhobar, ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # gives the characteristic radius at which the force from photo-ionized gas
    # is equal to that of the wind bubble, defined using the relationship 
    # Rch = Req^4 / RSt^3 and the above relations
    # no need to check for units as that is done above

    r_ch = Req(pdotw,rhobar, ci=ci)**4 / RSt(Q0, nbar,alphaB=alphaB)**3
    return r_ch.to("pc")

def Tion(nbar, alphaB=3.11e-13*(u.cm**3/u.s)):
    # gives the ionization-recombination time-scale in Myr
    # nbar : the number density of hydrgen in the background
    # alphaB : the case B recombination rate

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(nbar)=="number density"
    t2 = u.get_physical_type(alphaB)=="volumetric flow rate"
    if not(t1):
        print("Units of nbar are off")
        assert(False)
    if not(t2):
        print("Units of alphaB are off")
        assert(False)

    t_ion = (nbar*alphaB)**-1

    return t_ion.to("Myr")

def Tff(rhobar):
    # gives the free-fall timescale in Myr
    # rhobar : mass density

    t1 = u.get_physical_type(rhobar)=="mass density"
    if not(t1):
        print("Units of rhobar are off")
        assert(False)
    
    t_ff = (3*np.pi/(32*aconsts.G*rhobar))

    return t_ff.to("Myr")

def Teq(pdotw, rhobar, ci = 10*u.km/u.s):
    # gives the time it takes to reach Req
    # pdotw : wind momentum input rate
    # rhobar : background mass density
    # ci : ionized gas sound speed

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(pdotw)=="force"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        print("Units of pdotw are incorrect")
        assert(False)
    if not(t2):
        print("Units of rhobar are off")
        assert(False)
    if not(t3):
        print("Units of ci are off")
        assert(False)

    t_eq = (((3*pdotw/(2*np.pi*rhobar))**(1./2)))/(6*ci**2)

    return t_eq.to("Myr")

def TSt(Q0, nbar, pdotw, rhobar, ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # time at which an unimpeded wind bubble would reach the Stromgren Radius
    # parameters defined as above

    # get the stromgren radius
    r_st = RSt(Q0,nbar,alphaB=alphaB)

    t1 = u.get_physical_type(pdotw)=="force"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        print("Units of pdotw are incorrect")
        assert(False)
    if not(t2):
        print("Units of rhobar are off")
        assert(False)
    if not(t3):
        print("Units of ci are off")
        assert(False)
    
    t_st = (r_st**2)*(2*np.pi*rhobar/(3*pdotw))**(1./2)

    return t_st.to("Myr")

def Tdion(Q0, nbar, ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # the dynamical expansion time of an ionized gas bubble

    # get the stromgren radius
    r_st = RSt(Q0,nbar,alphaB=alphaB)

    t1 = u.get_physical_type(ci)=="speed"
    if not(t1):
        print("Units of ci are off")
        assert(False)
    
    t_di = np.sqrt(3)*r_st/(2*ci)

    return t_di.to("Myr")