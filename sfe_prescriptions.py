# Various prescriptions for the star formation efficiency (SFE)
# of a molecular cloud
# author: Lachlan Lancaster

from astropy import units as u


def estar_Grudic18(Sigma_cl, Scrit=2800*u.Msun/u.pc**2, emax=0.77):
    # sfe prescription from Equation 11 of Grudić et al. 2018
    # Sigma_cl : cloud surface density
    # Scrit : critical surface density
    # emax : maximum star formation efficiency parameter
    t1 = u.get_physical_type(Sigma_cl)=="surface mass density"
    t2 = u.get_physical_type(Scrit)=="surface mass density"
    if not(t1):
        print("Units of Sigma_cl are off")
        assert(False)
    if not(t2):
        print("Units of Scrit are off")
        assert(False)
    return 1./(1./emax + Scrit/Sigma_cl)

def pstar_mstar(Sigma_cl, prefac = 135*u.km/u.s):
    # momentum input per unit stellar mass as a funciton
    # of the cloud surface density
    # from Equation 18 of Kim et al. 2018
    # Sigma_cl : cloud surface density
    # prefac : prefactor of p/Mstar
    t1 = u.get_physical_type(Sigma_cl)=="surface mass density"
    t2 = u.get_physical_type(prefac)=="speed"
    if not(t1):
        print("Units of Sigma_cl are off")
        assert(False)
    if not(t2):
        print("Units of prefactor are off")
        assert(False)
    return prefac*(Sigma_cl/(100*u.Msun/u.pc**2))**-0.74

def estar_Kim18(Sigma_cl, vej=15*u.km/u.s,
                epsej=0.13, prefac=135*u.km/u.s):
    # sfe prescription from Equation 28 of Kim et al. 2018
    # Sigma_cl : cloud surface density
    # vej : average velocity of gas ejected from the cloud
    # epsej : fraction gas ejected in initial turbulence
    # prefac : prefactor of p/Mstar
    t1 = u.get_physical_type(Sigma_cl)=="surface mass density"
    t2 = u.get_physical_type(vej)=="speed"
    t3 = u.get_physical_type(epsej)=="dimensionless"
    t4 = u.get_physical_type(prefac)=="speed"
    if not(t1):
        print("Units of Sigma_cl are off")
        assert(False)
    if not(t2):
        print("Units of vej are off")
        assert(False)
    if not(t3):
        print("Units of epsej are off")
        assert(False)
    if not(t4):
        print("Units of prefac are off")
        assert(False)
    return (1- epsej)/(1 + pstar_mstar(Sigma_cl,prefac=prefac)/vej)