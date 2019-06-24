from numpy import exp, ma, maximum


def relative_humidity(p,q,t,A=17.625,B=-30.11,C=610.94,masked=False):
    """
    From Mark G. Lawrence, BAMS Feb 2005, eq. (6)

    RH = relative_humidity(p,q,t,A,B,C)
    
    inputs:   p = pressure (Pa)
              q = specific humidity (kg/kg)
              t = temperature (K)
    keywords: A, B and C are optional fitting parameters
              from Alduchov and Eskridge (1996).
              Masked = False (if True, perform operation on masked arrays)
    output:   RH = relative humidity (0-1)

    p, q and t can be arrays.
    """
    if masked==False:
        es = C * exp(A*(t-273.15)/(B+t))
        ws = 0.62198*es/(maximum(p,es)-(1-0.62198)*es)
        RH = q/ws
    else:
        es = C * ma.exp(A*(t-273.15)/(B+t))
        ws = 0.62198*es/(maximum(p,es)-(1-0.62198)*es)
        RH = q/ws
    return RH


def specific_humidity(p,RH,t,A=17.625,B=-30.11,C=610.94,masked=False):
    """
    From Mark G. Lawrence, BAMS Feb 2005, eq. (6)

    q = specific_humidity(p,RH,t,A,B,C)

    inputs:   p = pressure (Pa)
              RH = relative humidity (0-1)
              t = temperature (K)
    keywords: A, B and C are optional fitting parameters
              from Alduchov and Eskridge (1996).
              Masked = False (if True, perform operation on masked arrays)
    output:   q, specific humidity (kg/kg)

    p, RH and t can be arrays.
    """
    if masked==False:
        es = C * exp(A*(t-273.15)/(B+t))
        q = 0.62198*(RH*es)/(maximum(p,es)-(1-0.62198)*es)
    else:
        es = C * ma.exp(A*(t-273.15)/(B+t))
        q = 0.62198*(RH*es)/(maximum(p,es)-(1-0.62198)*es)
    return q

