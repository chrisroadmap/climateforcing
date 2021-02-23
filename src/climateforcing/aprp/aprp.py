import glob
import numpy as np
import warnings
from netCDF4 import Dataset

# TODO: allow scalar and non-3D arrays in the input
# TODO: docstrings for non-APRP function

def _pla(mu,gamma,alpha):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = mu * gamma + (mu*alpha*(1-gamma)**2)/(1-alpha*gamma)
        a[~np.isfinite(a)] = 0
    return a


def cloud_radiative_effect(base, pert):
    """Calculate the cloud radiative effect and approximate split of LW radiation
    into ERFari and ERFaci.

    Input:
        base, pert: dicts of CMIP diagnostics required to calculate APRP:
            'rlut'    : toa outgoing longwave flux
            'rlutcs'  : toa outgoing longwave flux assuming clear sky

    Output:
        ERFariLW, ERFaciLW: arrays
    """

    # check all required diagnostics are present
    check_vars = ['rlut', 'rlutcs']
    for var_dict in [base, pert]:
        for check_var in check_vars:
            if check_var not in var_dict.keys():
                raise ValueError('%s not present in %s' % (check_var, var_dict))
        var_dict['rlut'] = var_dict['rlut']
        var_dict['rlutcs'] = var_dict['rlutcs']

    ERFLW = -pert['rlut'] - (-base['rlut'])
    ERFaciLW = ERFLW - (-pert['rlutcs'] - (-base['rlutcs']))
    ERFariLW = ERFLW - ERFaciLW
    return ERFariLW, ERFaciLW


def aprp(base, pert, lw=False, breakdown=False, globalmean=False,
    lat=None, cs_threshold=0.02, clt_percent=True):
    """
    Approximate Partial Raditive Perturbation calculation

    This calculates the breakdown of shortwave radiative forcing into absorption
    and scattering components. When used with aerosol forcing, it can be used to
    separate the effective radiative forcing into aerosol-radiation (ERFari) and
    aerosol-cloud (ERFaci) components.

    References:
    -----------

    Zelinka, M. D., Andrews, T., Forster, P. M., and Taylor, K. E. (2014), Quantifying 
    components of aerosol‐cloud‐radiation interactions in climate models, J. Geophys.
    Res. Atmos., 119, 7599– 7615, https://doi.org/10.1002/2014JD021710.

    Taylor, K. E., Crucifix, M., Braconnot, P., Hewitt, C. D., Doutriaux, C., Broccoli,
    A. J., Mitchell, J. F. B., & Webb, M. J. (2007). Estimating Shortwave Radiative
    Forcing and Response in Climate Models, Journal of Climate, 20(11), 2530-2543,
    https://doi.org/10.1175/JCLI4143.1

    This implementation is a little different to Mark Zelinka's version at
    https://github.com/mzelinka/aprp.

    Input:
        base : dict
            baseline climate to use
        pert : dict
            perturbed climate to use
        lw : bool
            calculate the longwave forcing, in addition to the shortwave.
        breakdown : bool
            provide the forward and reverse calculations of APRP in the output, along
            with the central difference (the mean of forward and reverse)
        globalmean : bool
            provide global mean outputs (requires valid value for `lat`)
        lat : None or `numpy.ndarray`
            latitudes corresponding to axis numbered 1. Only required if globalmean
            is True.
        cs_threshold : float
            minimum cloud fraction (0-1 scale) for calculation of cloudy-sky APRP. If
            either perturbed or control run cloud fraction is below this, set the APRP
            flux to zero. It is recommended to use a small positive value, as the
            cloud fraction appears in the denominator of the calculation. Taken from 
            Mark Zelinka's implementation.
        clt_percent : bool
            is cloud fraction from base and pert in percent (True) or 0-1 scale
            (False)

    Both `base` and `pert` are `dict`s containing CMIP-style variables. They should be
    3-dimensional arrays in (time, latitude, longitude) format. The following items are
    required. CMIP variable naming conventions are used.

        rsdt    : TOA incoming shortwave flux (W m-2)
        rsus    : surface upwelling shortwave flux (W m-2)
        rsds    : surface downwelling_shortwave flux (W m-2)
        clt     : cloud area_fraction (fraction or %, see `clt_unit`)
        rsdscs  : surface downwelling shortwave flux assuming clear sky (W m-2)
        rsuscs  : surface upwelling shortwave flux assuming clear sky (W m-2)
        rsut    : TOA outgoing shortwave flux (W m-2)
        rsutcs  : TOA outgoing shortwave flux assuming clear sky (W m-2)

    If the longwave calculation is also required, the following should also be included

        rlut    : TOA outgoing longwave flux (W m-2)
        rlutcs  : TOA outgoing longwave flux assuming clear sky (W m-2)
          
    Returns:
        central[, forward, reverse] : dict(s)
            Components of APRP as defined by equation A2 of Zelinka et al., 2014

            dict elements are 't1', 't2', ..., 't9' where t? is the
            corresponding term in A2.

            't2_clr' and 't3_clr' are also provided, being hypothetical clear sky
            values of t2 and t3.

            Result dict(s) also contain 'ERFariSW', 'ERFaciSW' and 'albedo' where

            ERFariSW = t2 + t3 + t5 + t6
            ERFaciSW = t7 + t8 + t9
            albedo = t1 + t4
            ERFari_SWclr = t2_clr + t3_clr

            though note these only make sense if you are calculating aerosol forcing.
            The cloud fraction adjustment component of ERFaci is in t9.

            if lw==True, central also contains 'ERFariLW' and 'ERFaciLW' as
            calculated from the cloud radiative effect.
    """
    # check all required diagnostics are present
    if lw:
        check_vars = ['rsdt', 'rsus', 'rsds', 'clt', 'rsdscs', 'rsuscs',
                      'rsut', 'rsutcs', 'rlut', 'rlutcs']
    else:
        check_vars = ['rsdt', 'rsus', 'rsds', 'clt', 'rsdscs', 'rsuscs',
                      'rsut', 'rsutcs']
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

        # rescale cloud fraction to 0-1 if necessary
        if clt_percent:
            var_dict['clt'] = var_dict['clt']/100

    # require lat for globalmean
    if globalmean:
        if lat is None:
            raise ValueError('`lat` must be specified for `globalmean=True`')
        elif len(lat) != base['rsdt'].shape[1]:
            raise ValueError('`lat` must be the same length as axis 1 of input variables')

    # the catch_warnings stops divide by zeros being flagged
    # we might want to flag these after all and give user the option to disable
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
      
    dAoc_dacld = 0.5 * ( (_pla(mu_oc_base,gamma_oc_base,alpha_oc_pert)-_pla(mu_oc_base,gamma_oc_base,alpha_oc_base)) + (_pla(mu_oc_pert,gamma_oc_pert,alpha_oc_pert) - (_pla(mu_oc_pert,gamma_oc_pert,alpha_oc_base))) )
    dAoc_dmcld = 0.5 * ( (_pla(mu_pert,gamma_base,alpha_oc_base)-_pla(mu_base,gamma_base,alpha_oc_base)) + (_pla(mu_pert,gamma_pert,alpha_oc_pert) - (_pla(mu_base,gamma_pert,alpha_oc_pert))) )
    dAoc_dgcld = 0.5 * ( (_pla(mu_base,gamma_pert,alpha_oc_base)-_pla(mu_base,gamma_base,alpha_oc_base)) + (_pla(mu_pert,gamma_pert,alpha_oc_pert) - (_pla(mu_pert,gamma_base,alpha_oc_pert))) )

    dAoc_daaer = 0.5 * ( (_pla(mu_oc_base,gamma_oc_base,alpha_cs_pert)-_pla(mu_oc_base,gamma_oc_base,alpha_cs_base)) + (_pla(mu_oc_pert,gamma_oc_pert,alpha_cs_pert) - (_pla(mu_oc_pert,gamma_oc_pert,alpha_cs_base))) )
    dAoc_dmaer = 0.5 * ( (_pla(mu_cs_pert,gamma_oc_base,alpha_oc_base)-_pla(mu_cs_base,gamma_oc_base,alpha_oc_base)) + (_pla(mu_cs_pert,gamma_oc_pert,alpha_oc_pert) - (_pla(mu_cs_base,gamma_oc_pert,alpha_oc_pert))) )
    dAoc_dgaer = 0.5 * ( (_pla(mu_oc_base,gamma_cs_pert,alpha_oc_base)-_pla(mu_oc_base,gamma_cs_base,alpha_oc_base)) + (_pla(mu_oc_pert,gamma_cs_pert,alpha_oc_pert) - (_pla(mu_oc_pert,gamma_cs_base,alpha_oc_pert))) )

    dAcs_daclr = 0.5 * ( (_pla(mu_cs_base,gamma_cs_base,alpha_cs_pert)-_pla(mu_cs_base,gamma_cs_base,alpha_cs_base)) + (_pla(mu_cs_pert,gamma_cs_pert,alpha_cs_pert) - (_pla(mu_cs_pert,gamma_cs_pert,alpha_cs_base))) )
    dAcs_dmaer = 0.5 * ( (_pla(mu_cs_pert,gamma_cs_base,alpha_cs_base)-_pla(mu_cs_base,gamma_cs_base,alpha_cs_base)) + (_pla(mu_cs_pert,gamma_cs_pert,alpha_cs_pert) - (_pla(mu_cs_base,gamma_cs_pert,alpha_cs_pert))) )
    dAcs_dgaer = 0.5 * ( (_pla(mu_cs_base,gamma_cs_pert,alpha_cs_base)-_pla(mu_cs_base,gamma_cs_base,alpha_cs_base)) + (_pla(mu_cs_pert,gamma_cs_pert,alpha_cs_pert) - (_pla(mu_cs_pert,gamma_cs_base,alpha_cs_pert))) )

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
    forward['t2_clr'] = -base['rsntcs']*dAcs_dgaer
    forward['t3_clr'] = -base['rsntcs']*dAcs_dmaer
    
    # set thresholds
    # TODO: can we avoid a hard cloud fraction threshold here?
    forward['t4'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., forward['t4'])
    forward['t5'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., forward['t5'])
    forward['t6'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., forward['t6'])
    forward['t7'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., forward['t7'])
    forward['t8'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., forward['t8'])
    forward['t9'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., forward['t9'])

    forward['ERFariSWclr'] = forward['t2_clr'] + forward['t3_clr']
    forward['ERFariSW'] = forward['t2'] + forward['t3'] + forward['t5'] + forward['t6']
    forward['ERFaciSW'] = forward['t7'] + forward['t8'] + forward['t9']
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
    reverse['t2_clr'] = -pert['rsntcs']*dAcs_dgaer
    reverse['t3_clr'] = -pert['rsntcs']*dAcs_dmaer

    # set thresholds
    reverse['t4'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., reverse['t4'])
    reverse['t5'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., reverse['t5'])
    reverse['t6'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., reverse['t6'])
    reverse['t7'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., reverse['t7'])
    reverse['t8'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., reverse['t8'])
    reverse['t9'] = np.where(np.logical_or(base['clt']<cs_threshold, pert['clt']<cs_threshold), 0., reverse['t9'])

    reverse['ERFariSWclr'] = reverse['t2_clr'] + reverse['t3_clr']
    reverse['ERFariSW'] = reverse['t2'] + reverse['t3'] + reverse['t5'] + reverse['t6']
    reverse['ERFaciSW'] = reverse['t7'] + reverse['t8'] + reverse['t9']
    reverse['albedo'] = reverse['t1'] + reverse['t4']

    for key in forward.keys():
        central[key] = 0.5 * (forward[key] + reverse[key])

    if lw:
        central['ERFariLW'], central['ERFaciLW'] = cloud_radiative_effect(base, pert)
        forward['ERFariLW'] = np.copy(central['ERFariLW'])
        reverse['ERFariLW'] = np.copy(central['ERFariLW'])
        forward['ERFaciLW'] = np.copy(central['ERFaciLW'])
        reverse['ERFaciLW'] = np.copy(central['ERFaciLW'])

    if globalmean:
        # is latitude ascending or descending?
        if lat[0] < lat[-1]:
            latbounds = np.concatenate(([-90], 0.5*lat[1:]+lat[:-1], [90]))
            weights = np.diff(np.sin(np.radians(latbounds)))[None,:,None] * np.ones((base['rsdt'].shape[0], 1, base['rsdt'].shape[2]))
        else:
            latbounds = np.concatenate(([90], 0.5*lat[1:]+lat[:-1], [-90]))
            weights = -np.diff(np.sin(np.radians(latbounds)))[None,:,None] * np.ones((base['rsdt'].shape[0], 1, base['rsdt'].shape[2]))
        for key in central.keys():
            central[key] = np.average(central[key], weights=weights)
            forward[key] = np.average(forward[key], weights=weights)
            reverse[key] = np.average(reverse[key], weights=weights)

    if breakdown:
        return central, forward, reverse
    else:
        return central


def create_input(basedir, pertdir, latout=False, lw=False, 
    sl=slice(0,None,None)):
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
        latout: True if lat variable to be included in output.
        lw: Set to True to do LW calculation using CRE.
        sl: slice of indices to use from each dataset.

    Outputs:
        base: dict of variables needed for APRP from control
        pert: dict of variables needed for APRP from experiment
        [lat]: latitude points relating to axis 1 of arrays
    """

    base = {}
    pert = {}

    if lw:
        varlist=['rsdt', 'rsus', 'rsds', 'clt', 'rsdscs', 'rsuscs', 'rsut', 
                 'rsutcs', 'rlut', 'rlutcs']
    else:
        varlist=['rsdt', 'rsus', 'rsds', 'clt', 'rsdscs', 'rsuscs', 'rsut',
                 'rsutcs']

    for var in varlist:
        filenames = sorted(glob.glob('%s/%s_*.nc' % (basedir, var)))
        if len(filenames)==0:
            raise RuntimeError('No variables of name %s found in directory %s'
                % (var, basedir))
        for i, filename in enumerate(filenames):
            nc = Dataset(filename)
            invar = nc.variables[var][sl,...]
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
            invar = nc.variables[var][sl,...]
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
