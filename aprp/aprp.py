from __future__ import division

import glob
import numpy as np
import warnings
from netCDF4 import Dataset

def pla(mu,gamma,alpha):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = mu * gamma + (mu*alpha*(1-gamma)**2)/(1-alpha*gamma)
        a[~np.isfinite(a)] = 0
    return a


def calc_aprp(base, pert, breakdown=False, globalmean=False, lat=None):
    """Calculate fluxes using the Approximate Radiative Perturbation method
    (Taylor et al., 2007, https://journals.ametsoc.org/doi/pdf/10.1175/JCLI4143.1)
  
    Input:
        base, pert: dicts of CMIP diagnostics required to calculate APRP:
            'rsdt'    : toa incoming shortwave flux
            'rsus'    : surface upwelling shortwave flux
            'rsds'    : surface downwelling_shortwave flux
            'clt'     : cloud area_fraction
            'rsdscs'  : surface downwelling shortwave flux assuming clear sky
            'rsuscs'  : surface upwelling shortwave flux assuming clear sky
            'rsut'    : toa outgoing shortwave flux
            'rsutcs'  : toa outgoing shortwave flux assuming clear sky
    
    Keyword:
        breakdown: if True, provide the forward and reverse calculations of
            APRP in the output, along with the central difference (the mean of
            forward and reverse)
        globalmean: if True, calculate global mean diagnostics (else do
            gridpoint by gridpoint). If globalmean=True, lat must be specified
        lat: latitudes of axis 1. Only required if globalmean=True
          
    Output:
        central[, forward, reverse]: dict(s) of components of APRP as defined
            by equation A2 of Zelinka et al., 2014
            https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2014JD021710

            dict elements are 't1', 't2', ..., 't9' where t? is the
            corresponding term in A2.

            dict(s) also contain 'ERFari', 'ERFaci' and 'albedo' where

            ERFari = t2 + t3 + t5 + t6
            ERFaci = t7 + t8 + t9
            albedo = t1 + t4
    """


    # check all required diagnostics are present
    check_vars = ['rsdt', 'rsus', 'rsds', 'clt', 'rsdscs', 'rsuscs', 'rsut', 'rsutcs']
    for var_dict in [base, pert]:
        for check_var in check_vars:
            if check_var not in var_dict.keys():
                raise ValueError('%s not present in %s' % (check_var, var_dict))
            if var_dict[check_var].ndim != 3:
                raise ValueError('%s in %s has %d dimensions (should be 3)' %
                    (check_var, var_dict, var_dict[check_var].ndim))
            # if we get to here, rsdt exists, so verify all diagnostics have same shape
            if var_dict[check_var].shape != var_dict['rsdt'].shape:
                raise ValueError('%s %s in %s differs in shape to rsdt %s' %
                    (check_var, var_dict[check_var].shape, var_dict, var_dict['rsdt'].shape))
        # rescale cloud fraction to 0-1
        var_dict['clt'] = var_dict['clt']/100.

    # require lat for globalmean
    if globalmean:
        if lat is None:
            raise ValueError('lat must be specified for global mean calculation')
        elif len(lat) != base['rsdt'].shape[1]:
            raise ValueError('lat must be the same length as axis 1 of input variables')

    # the catch_warnings stops divide by zeros being flagged
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base['rsutoc'] = (base['rsut'] - (1 - base['clt'])*(base['rsutcs']))/base['clt']
        pert['rsutoc'] = (pert['rsut'] - (1 - pert['clt'])*(pert['rsutcs']))/pert['clt']
        base['rsutoc'][~np.isfinite(base['rsutoc'])] = pert['rsutoc'][~np.isfinite(base['rsutoc'])]
        pert['rsutoc'][~np.isfinite(pert['rsutoc'])] = base['rsutoc'][~np.isfinite(pert['rsutoc'])]
        base['rsutoc'][~np.isfinite(base['rsutoc'])] = base['rsutcs'][~np.isfinite(base['rsutoc'])]
        pert['rsutoc'][~np.isfinite(pert['rsutoc'])] = pert['rsutcs'][~np.isfinite(pert['rsutoc'])]
        rsutoc = 0.5*(pert['rsutoc'] + base['rsutoc'])
        rsutcs = 0.5*(pert['rsutcs'] + base['rsutcs'])
        delta_clt = pert['clt'] - base['clt']
        clt = 0.5*(pert['clt'] + base['clt'])
        delta_rsutoc = pert['rsutoc'] - base['rsutoc']
        delta_rsutcs = pert['rsutcs'] - base['rsutcs']

        base['rsusoc'] = (base['rsus'] - (1 - base['clt'])*(base['rsuscs']))/base['clt']
        pert['rsusoc'] = (pert['rsus'] - (1 - pert['clt'])*(pert['rsuscs']))/pert['clt']
        base['rsusoc'][~np.isfinite(base['rsusoc'])] = pert['rsusoc'][~np.isfinite(base['rsusoc'])]
        pert['rsusoc'][~np.isfinite(pert['rsusoc'])] = base['rsusoc'][~np.isfinite(pert['rsusoc'])]
        base['rsusoc'][~np.isfinite(base['rsusoc'])] = base['rsuscs'][~np.isfinite(base['rsusoc'])]
        pert['rsusoc'][~np.isfinite(pert['rsusoc'])] = pert['rsuscs'][~np.isfinite(pert['rsusoc'])]
        rsusoc = 0.5*(pert['rsusoc'] + base['rsusoc'])
        rsuscs = 0.5*(pert['rsuscs'] + base['rsuscs'])
        delta_rsusoc = pert['rsusoc'] - base['rsusoc']
        delta_rsuscs = pert['rsuscs'] - base['rsuscs']

        base['rsdsoc']  = (base['rsds'] - (1 - base['clt'])*(base['rsdscs']))/base['clt']
        pert['rsdsoc']  = (pert['rsds'] - (1 - pert['clt'])*(pert['rsdscs']))/pert['clt']
        base['rsdsoc'][~np.isfinite(base['rsdsoc'])] = pert['rsdsoc'][~np.isfinite(base['rsdsoc'])]
        pert['rsdsoc'][~np.isfinite(pert['rsdsoc'])] = base['rsdsoc'][~np.isfinite(pert['rsdsoc'])]
        base['rsdsoc'][~np.isfinite(base['rsdsoc'])] = base['rsdscs'][~np.isfinite(base['rsdsoc'])]
        pert['rsdsoc'][~np.isfinite(pert['rsdsoc'])] = pert['rsdscs'][~np.isfinite(pert['rsdsoc'])]
        rsdsoc = 0.5*(pert['rsdsoc'] + base['rsdsoc'])
        rsdscs = 0.5*(pert['rsdscs'] + base['rsdscs'])
        delta_rsdsoc = pert['rsdsoc'] - base['rsdsoc']
        delta_rsdscs = pert['rsdscs'] - base['rsdscs']
      
        A_oc_base = base['rsutoc']/base['rsdt']
        A_oc_base[~np.isfinite(A_oc_base)] = 0.  # this is safe
        alpha_oc_base = base['rsusoc']/base['rsdsoc']
        alpha_oc_base[~np.isfinite(alpha_oc_base)] = 0.
        Q_oc_down_base = base['rsdsoc']/base['rsdt']
        Q_oc_down_base[~np.isfinite(Q_oc_down_base)] = 0.
        mu_oc_base = A_oc_base + Q_oc_down_base*(1-alpha_oc_base)
        gamma_oc_base = (mu_oc_base - Q_oc_down_base)/(mu_oc_base - alpha_oc_base*Q_oc_down_base)
        gamma_oc_base[~np.isfinite(gamma_oc_base)] = 0.

        A_oc_pert = pert['rsutoc']/pert['rsdt']
        A_oc_pert[~np.isfinite(A_oc_pert)] = 0.
        alpha_oc_pert = pert['rsusoc']/pert['rsdsoc']
        alpha_oc_pert[~np.isfinite(alpha_oc_pert)] = 0.
        Q_oc_down_pert = pert['rsdsoc']/pert['rsdt']
        Q_oc_down_pert[~np.isfinite(Q_oc_down_pert)] = 0.
        mu_oc_pert = A_oc_pert + Q_oc_down_pert*(1-alpha_oc_pert)
        gamma_oc_pert = (mu_oc_pert - Q_oc_down_pert)/(mu_oc_pert - alpha_oc_pert*Q_oc_down_pert)
        gamma_oc_pert[~np.isfinite(gamma_oc_pert)] = 0.

        A_cs_base = base['rsutcs']/base['rsdt']
        A_cs_base[~np.isfinite(A_cs_base)] = 0.
        alpha_cs_base = base['rsuscs']/base['rsdscs']
        alpha_cs_base[~np.isfinite(alpha_cs_base)] = 0.
        Q_cs_down_base = base['rsdscs']/base['rsdt']
        Q_cs_down_base[~np.isfinite(Q_cs_down_base)] = 0.
        mu_cs_base = A_cs_base + Q_cs_down_base*(1-alpha_cs_base)
        gamma_cs_base = (mu_cs_base - Q_cs_down_base)/(mu_cs_base - alpha_cs_base*Q_cs_down_base)
        gamma_cs_base[~np.isfinite(gamma_cs_base)] = 0.

        A_cs_pert = pert['rsutcs']/pert['rsdt']
        A_cs_pert[~np.isfinite(A_cs_pert)] = 0.
        alpha_cs_pert = pert['rsuscs']/pert['rsdscs']
        alpha_cs_pert[~np.isfinite(alpha_cs_pert)] = 0.
        Q_cs_down_pert = pert['rsdscs']/pert['rsdt']
        Q_cs_down_pert[~np.isfinite(Q_cs_down_pert)] = 0.
        mu_cs_pert = A_cs_pert + Q_cs_down_pert*(1-alpha_cs_pert)
        gamma_cs_pert = (mu_cs_pert - Q_cs_down_pert)/(mu_cs_pert - alpha_cs_pert*Q_cs_down_pert)
        gamma_cs_pert[~np.isfinite(gamma_cs_pert)] = 0.

        # Calculate cloudy values of gamma and mu
        gamma_pert = 1 - (1 - gamma_oc_pert)/(1-gamma_cs_pert)
        mu_pert = (mu_oc_pert)/mu_cs_pert
        mu_pert[~np.isfinite(mu_pert)] = 0.
        gamma_base = 1 - (1 - gamma_oc_base)/(1-gamma_cs_base)
        mu_base = (mu_oc_base)/mu_cs_base
        mu_base[~np.isfinite(mu_base)] = 0.
      
    dAoc_dacld = 0.5 * ( (pla(mu_oc_base,gamma_oc_base,alpha_oc_pert)-pla(mu_oc_base,gamma_oc_base,alpha_oc_base)) + (pla(mu_oc_pert,gamma_oc_pert,alpha_oc_pert) - (pla(mu_oc_pert,gamma_oc_pert,alpha_oc_base))) )
    dAoc_dmcld = 0.5 * ( (pla(mu_pert,gamma_base,alpha_oc_base)-pla(mu_base,gamma_base,alpha_oc_base)) + (pla(mu_pert,gamma_pert,alpha_oc_pert) - (pla(mu_base,gamma_pert,alpha_oc_pert))) )
    dAoc_dgcld = 0.5 * ( (pla(mu_base,gamma_pert,alpha_oc_base)-pla(mu_base,gamma_base,alpha_oc_base)) + (pla(mu_pert,gamma_pert,alpha_oc_pert) - (pla(mu_pert,gamma_base,alpha_oc_pert))) )

    dAoc_daaer = 0.5 * ( (pla(mu_oc_base,gamma_oc_base,alpha_cs_pert)-pla(mu_oc_base,gamma_oc_base,alpha_cs_base)) + (pla(mu_oc_pert,gamma_oc_pert,alpha_cs_pert) - (pla(mu_oc_pert,gamma_oc_pert,alpha_cs_base))) )
    dAoc_dmaer = 0.5 * ( (pla(mu_cs_pert,gamma_oc_base,alpha_oc_base)-pla(mu_cs_base,gamma_oc_base,alpha_oc_base)) + (pla(mu_cs_pert,gamma_oc_pert,alpha_oc_pert) - (pla(mu_cs_base,gamma_oc_pert,alpha_oc_pert))) )
    dAoc_dgaer = 0.5 * ( (pla(mu_oc_base,gamma_cs_pert,alpha_oc_base)-pla(mu_oc_base,gamma_cs_base,alpha_oc_base)) + (pla(mu_oc_pert,gamma_cs_pert,alpha_oc_pert) - (pla(mu_oc_pert,gamma_cs_base,alpha_oc_pert))) )

    dAcs_daclr = 0.5 * ( (pla(mu_cs_base,gamma_cs_base,alpha_cs_pert)-pla(mu_cs_base,gamma_cs_base,alpha_cs_base)) + (pla(mu_cs_pert,gamma_cs_pert,alpha_cs_pert) - (pla(mu_cs_pert,gamma_cs_pert,alpha_cs_base))) )
    dAcs_dmaer = 0.5 * ( (pla(mu_cs_pert,gamma_cs_base,alpha_cs_base)-pla(mu_cs_base,gamma_cs_base,alpha_cs_base)) + (pla(mu_cs_pert,gamma_cs_pert,alpha_cs_pert) - (pla(mu_cs_base,gamma_cs_pert,alpha_cs_pert))) )
    dAcs_dgaer = 0.5 * ( (pla(mu_cs_base,gamma_cs_pert,alpha_cs_base)-pla(mu_cs_base,gamma_cs_base,alpha_cs_base)) + (pla(mu_cs_pert,gamma_cs_pert,alpha_cs_pert) - (pla(mu_cs_pert,gamma_cs_base,alpha_cs_pert))) )

    base['rsnt'] = base['rsdt']-base['rsut']
    base['rsntcs'] = base['rsdt']-base['rsutcs']
    pert['rsnt']  = pert['rsdt']-pert['rsut']
    pert['rsntcs'] = pert['rsdt']-pert['rsutcs']
    rsnt = 0.5*(base['rsnt']+pert['rsnt'])
    rsntcs = 0.5*(base['rsntcs']+pert['rsntcs'])
    pert['rsntoc'] = pert['rsdt']-pert['rsutoc']
    base['rsntoc'] = base['rsdt']-base['rsutoc']
    rsntoc = 0.5*(pert['rsntoc']+base['rsntoc'])
  
    # t1 to t9 are the coefficients of equation A2 in Zelinka et al., 2014
    forward = {}
    reverse = {}
    central = {}
    forward['t1'] = -base['rsntcs']*(1-clt)*dAcs_daclr
    forward['t2'] = -base['rsntcs']*(1-clt)*dAcs_dgaer
    forward['t3'] = -base['rsntcs']*(1-clt)*dAcs_dmaer
    forward['t4'] = -base['rsnt']*clt*(dAoc_dacld)
    forward['t5'] = -base['rsnt']*clt*(dAoc_dgaer)
    forward['t6'] = -base['rsnt']*clt*(dAoc_dmaer)
    forward['t7'] = -base['rsnt']*clt*(dAoc_dgcld)
    forward['t8'] = -base['rsnt']*clt*(dAoc_dmcld)
    forward['t9'] = -delta_clt * (rsutoc - rsutcs)
    forward['ERFari'] = forward['t2'] + forward['t3'] + forward['t5'] + forward['t6']
    forward['ERFaci'] = forward['t7'] + forward['t8'] + forward['t9']
    forward['albedo'] = forward['t1'] + forward['t4']

    reverse['t1'] = -pert['rsntcs']*(1-clt)*dAcs_daclr
    reverse['t2'] = -pert['rsntcs']*(1-clt)*dAcs_dgaer
    reverse['t3'] = -pert['rsntcs']*(1-clt)*dAcs_dmaer
    reverse['t4'] = -pert['rsnt']*clt*(dAoc_dacld)
    reverse['t5'] = -pert['rsnt']*clt*(dAoc_dgaer)
    reverse['t6'] = -pert['rsnt']*clt*(dAoc_dmaer)
    reverse['t7'] = -pert['rsnt']*clt*(dAoc_dgcld)
    reverse['t8'] = -pert['rsnt']*clt*(dAoc_dmcld)
    reverse['t9'] = -delta_clt * (rsutoc - rsutcs)
    reverse['ERFari'] = reverse['t2'] + reverse['t3'] + reverse['t5'] + reverse['t6']
    reverse['ERFaci'] = reverse['t7'] + reverse['t8'] + reverse['t9']
    reverse['albedo'] = reverse['t1'] + reverse['t4']

    for key in forward.keys():
        central[key] = 0.5 * (forward[key] + reverse[key])

    if globalmean:
        # is latitude ascending or descending?
        if lat[0] < lat[-1]:
            latbounds = np.concatenate(([-90], 0.5*lat[1:]+lat[:-1], [90]))
            weights = np.diff(np.sin(np.radians(latbounds)))[None,:,None] * np.ones((base['rsdt'].shape[0], 1, base['rsdt'].shape[2]))
        else:
            latbounds = np.concatenate(([90], 0.5*lat[1:]+lat[:-1], [-90]))
            weights = -np.diff(np.sin(np.radians(latbounds)))[None,:,None] * np.ones((base['rsdt'].shape[0], 1, base['rsdt'].shape[2]))
        for key in forward.keys():
            central[key] = np.average(central[key], weights=weights)
            forward[key] = np.average(forward[key], weights=weights)
            reverse[key] = np.average(reverse[key], weights=weights)

    if breakdown:
        return central, forward, reverse
    else:
        return central


def create_input(basedir, pertdir, latout=False):
    """Utility function to extract variables from a given directory and place
     into dictionaries.

    It assumes that base and pert are different directories and only one 
    experiment output is present in each directory.

    Slicing into time chunks is allowed and providing the filenames
    follow CMIP6 convention they should be concatenated in the correct
    order.

    Variables required are rsdt, rsus, rsds, clt, rsdscs, rsuscs, rsut, rsutcs
    An error will be raised if variables are not detected.

    Inputs:
        basedir: directory containing control variables
        pertdir: directory containing forcing perturbed variables

    Keywords:
        latout: True if lat variable to be included in output.

    Outputs:
        base: dict of 8 variables needed for APRP from control
        pert: dict of 8 variables needed for APRP from experiment
        [lat]: latitude points relating to axis 1 of arrays
    """

    base = {}
    pert = {}

    for var in ['rsdt', 'rsus', 'rsds', 'clt', 'rsdscs', 'rsuscs', 'rsut', 'rsutcs']:
        filenames = sorted(glob.glob('%s/%s_*.nc' % (basedir, var)))
        if len(filenames)==0:
            raise RuntimeError('No variables of name %s found in directory %s'
                % (var, basedir))
        for i, filename in enumerate(filenames):
            nc = Dataset(filename)
            invar = nc.variables[var][:]
            lat = nc.variables['lat'][:]
            nc.close()
            if i==0:
                base[var] = invar
            else:
                base[var] = np.append(base[var], invar, axis=0)

        filenames = sorted(glob.glob('%s/%s_*.nc' % (pertdir, var)))
        if len(filenames)==0:
            raise RuntimeError('No variables of name %s found in directory %s'
                % (var, pertdir))
        for i, filename in enumerate(filenames):
            nc = Dataset(filename)
            invar = nc.variables[var][:]
            lat = nc.variables['lat'][:]
            nc.close()
            if i==0:
                pert[var] = invar
            else:
                pert[var] = np.append(pert[var], invar, axis=0)

    if latout:
        return base, pert, lat
    else:
        return base, pert
