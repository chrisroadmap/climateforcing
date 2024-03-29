"""APRP: the approximate partial radiative perturbation calculation."""

# Modified Feb 2023 by M. Zelinka to do the following:
# 1) Multiply albedo sensitivities by downwelling SW rather than net (down minus up) SW
# 2) Fix the manner of estimating overcast albedo sensitivity to cloud and non-cloud
#    gamma and mu
# 3) Separate the forward and backward albedo sensitivity calculations

# These modifications lead to negligible TOA SW residuals and to
# perfect agreement with the results of Mark's code (https://github.com/mzelinka/aprp)

import glob
import warnings

import numpy as np
from netCDF4 import Dataset

# TODO: sort out refactoring for main aprp function


def _planetary_albedo(mu, gamma, alpha):  # pylint: disable=invalid-name
    """Planetary albedo, eq. (7) of Taylor et al. (2007)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pla = mu * gamma + (mu * alpha * (1 - gamma) ** 2) / (1 - alpha * gamma)
        pla[~np.isfinite(pla)] = 0
    return pla


def cloud_radiative_effect(base, pert):
    """Calculate the cloud radiative effect intended for longwave fluxes.

    Parameters
    ----------
    base, pert: dict of array_like
        CMIP diagnostics required to calculate longwave cloud radiative effect. The
        dicts should contain two keys:
        rlut    : top-of-atmosphere outgoing longwave flux
        rlutcs  : top-of-atmosphere longwave flux assuming clear sky
    Returns
    -------
    erfari_lw, erfaci_lw : array_like
        Longwave ERFari and ERFaci estimates.
    """
    # check all required diagnostics are present
    check_vars = ["rlut", "rlutcs"]
    for var_dict in [base, pert]:
        for check_var in check_vars:
            if check_var not in var_dict.keys():
                raise ValueError(f"{check_var} not present in {var_dict}")
        var_dict["rlut"] = var_dict["rlut"]
        var_dict["rlutcs"] = var_dict["rlutcs"]

    erf_lw = -pert["rlut"] - (-base["rlut"])
    erfaci_lw = erf_lw - (-pert["rlutcs"] - (-base["rlutcs"]))
    erfari_lw = erf_lw - erfaci_lw
    return erfari_lw, erfaci_lw


def aprp(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-branches  # noqa: E501
    base,
    pert,
    longwave=False,
    breakdown=False,
    cs_threshold=0.02,
    rsdt_threshold=0.1,
    clt_percent=True,
):
    """Approximate Partial Raditive Perturbation calculation.

    This calculates the breakdown of shortwave radiative forcing into absorption
    and scattering components. When used with aerosol forcing, it can be used to
    separate the effective radiative forcing into aerosol-radiation (ERFari) and
    aerosol-cloud (ERFaci) components.

    Parameters
    ----------
        base, pert : dict of array_like
            Baseline and perturbation climate states to use.

            Both `base` and `pert` are dicts containing CMIP-style variables. CMIP
            variable naming conventions are used. The dicts should contain the
            following keys:

            rsdt    : TOA incoming shortwave flux (W m-2)
            rsus    : surface upwelling shortwave flux (W m-2)
            rsds    : surface downwelling_shortwave flux (W m-2)
            clt     : cloud area_fraction (fraction or %, see `clt_unit`)
            rsdscs  : surface downwelling shortwave flux assuming clear sky (W m-2)
            rsuscs  : surface upwelling shortwave flux assuming clear sky (W m-2)
            rsut    : TOA outgoing shortwave flux (W m-2)
            rsutcs  : TOA outgoing shortwave flux assuming clear sky (W m-2)

            If the longwave calculation is also required, the following keys should
            also be included:

            rlut    : TOA outgoing longwave flux (W m-2)
            rlutcs  : TOA outgoing longwave flux assuming clear sky (W m-2)

        longwave : bool, default=True
            calculate the longwave forcing, in addition to the shortwave.
        breakdown : bool, default=False
            provide the forward and reverse calculations of APRP in the output, along
            with the central difference (the mean of forward and reverse)
        cs_threshold : float, default=0.02
            minimum cloud fraction (0-1 scale) for calculation of cloudy-sky APRP. If
            either perturbed or control run cloud fraction is below this, set the APRP
            flux to zero. It is recommended to use a small positive value, as the
            cloud fraction appears in the denominator of the calculation. Taken from
            Mark Zelinka's implementation.
        rsdt_threshold : float, default=0.1
            set incoming radiation to zero below a certain level. A small positive value
            is recommended to reduce residuals of the SW component compared to TOA ERF.
        clt_percent : bool, default=True
            is cloud fraction from base and pert in percent (True) or 0-1 scale
            (False)

    Returns
    -------
        central[, forward, reverse] : dict of array_like

            Components of APRP as defined by equation A2 of [1]_.
            dict keys are 't1', 't2', ..., 't9' where tX is the corresponding term in
            eq. A2.

            't2_clr' and 't3_clr' are also provided, being hypothetical clear sky values
            of t2 and t3.

            Result dict(s) also contain 'ERFariSW', 'ERFaciSW' and 'albedo' where

            ERFariSW = t2 + t3 + t5 + t6
            ERFaciSW = t7 + t8 + t9
            albedo = t1 + t4
            ERFari_SWclr = t2_clr + t3_clr

            though note these only make sense if you are calculating aerosol forcing.
            The cloud fraction adjustment component of ERFaci is t9.
            if longwave is True, central also contains 'ERFariLW' and 'ERFaciLW' as
            calculated from the cloud radiative effect.

    Notes
    -----
    This implementation is a little different to Mark Zelinka's version at
    https://github.com/mzelinka/aprp.

    References
    ----------
    .. [1] Zelinka, M. D., Andrews, T., Forster, P. M., and Taylor, K. E. (2014),
    Quantifying components of aerosol‐cloud‐radiation interactions in climate
    models, J. Geophys. Res. Atmos., 119, 7599– 7615,
    https://doi.org/10.1002/2014JD021710.

    .. [2] Taylor, K. E., Crucifix, M., Braconnot, P., Hewitt, C. D., Doutriaux, C.,
    Broccoli, A. J., Mitchell, J. F. B., & Webb, M. J. (2007). Estimating Shortwave
    Radiative Forcing and Response in Climate Models, Journal of Climate, 20(11),
    2530-2543, https://doi.org/10.1175/JCLI4143.1
    """
    # check all required diagnostics are present
    if longwave:
        check_vars = [
            "rsdt",
            "rsus",
            "rsds",
            "clt",
            "rsdscs",
            "rsuscs",
            "rsut",
            "rsutcs",
            "rlut",
            "rlutcs",
        ]
    else:
        check_vars = [
            "rsdt",
            "rsus",
            "rsds",
            "clt",
            "rsdscs",
            "rsuscs",
            "rsut",
            "rsutcs",
        ]
    for var_dict in [base, pert]:
        for check_var in check_vars:
            if check_var not in var_dict.keys():
                raise ValueError(f"{check_var} not present in {var_dict}")
            # if we get to here, rsdt exists, so verify all diagnostics have same shape
            if var_dict[check_var].shape != var_dict["rsdt"].shape:
                raise ValueError(
                    f"{check_var} {var_dict[check_var].shape} in {var_dict} "
                    f"differs in shape to rsdt {var_dict['rsdt'].shape}"
                )

        # rescale cloud fraction to 0-1 if necessary
        if clt_percent:
            var_dict["clt"] = var_dict["clt"] / 100

    # the catch_warnings stops divide by zeros being flagged
    # we might want to flag these after all and give user the option to disable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base["rsutoc"] = (base["rsut"] - (1 - base["clt"]) * (base["rsutcs"])) / base[
            "clt"
        ]
        pert["rsutoc"] = (pert["rsut"] - (1 - pert["clt"]) * (pert["rsutcs"])) / pert[
            "clt"
        ]
        base["rsutoc"][~np.isfinite(base["rsutoc"])] = pert["rsutoc"][
            ~np.isfinite(base["rsutoc"])
        ]
        pert["rsutoc"][~np.isfinite(pert["rsutoc"])] = base["rsutoc"][
            ~np.isfinite(pert["rsutoc"])
        ]
        base["rsutoc"][~np.isfinite(base["rsutoc"])] = base["rsutcs"][
            ~np.isfinite(base["rsutoc"])
        ]
        pert["rsutoc"][~np.isfinite(pert["rsutoc"])] = pert["rsutcs"][
            ~np.isfinite(pert["rsutoc"])
        ]
        rsutoc = 0.5 * (pert["rsutoc"] + base["rsutoc"])
        rsutcs = 0.5 * (pert["rsutcs"] + base["rsutcs"])
        delta_clt = pert["clt"] - base["clt"]
        rsdt = 0.5 * (base["rsdt"] + pert["rsdt"])

        base["rsusoc"] = (base["rsus"] - (1 - base["clt"]) * (base["rsuscs"])) / base[
            "clt"
        ]
        pert["rsusoc"] = (pert["rsus"] - (1 - pert["clt"]) * (pert["rsuscs"])) / pert[
            "clt"
        ]
        base["rsusoc"][~np.isfinite(base["rsusoc"])] = pert["rsusoc"][
            ~np.isfinite(base["rsusoc"])
        ]
        pert["rsusoc"][~np.isfinite(pert["rsusoc"])] = base["rsusoc"][
            ~np.isfinite(pert["rsusoc"])
        ]
        base["rsusoc"][~np.isfinite(base["rsusoc"])] = base["rsuscs"][
            ~np.isfinite(base["rsusoc"])
        ]
        pert["rsusoc"][~np.isfinite(pert["rsusoc"])] = pert["rsuscs"][
            ~np.isfinite(pert["rsusoc"])
        ]

        base["rsdsoc"] = (base["rsds"] - (1 - base["clt"]) * (base["rsdscs"])) / base[
            "clt"
        ]
        pert["rsdsoc"] = (pert["rsds"] - (1 - pert["clt"]) * (pert["rsdscs"])) / pert[
            "clt"
        ]
        base["rsdsoc"][~np.isfinite(base["rsdsoc"])] = pert["rsdsoc"][
            ~np.isfinite(base["rsdsoc"])
        ]
        pert["rsdsoc"][~np.isfinite(pert["rsdsoc"])] = base["rsdsoc"][
            ~np.isfinite(pert["rsdsoc"])
        ]
        base["rsdsoc"][~np.isfinite(base["rsdsoc"])] = base["rsdscs"][
            ~np.isfinite(base["rsdsoc"])
        ]
        pert["rsdsoc"][~np.isfinite(pert["rsdsoc"])] = pert["rsdscs"][
            ~np.isfinite(pert["rsdsoc"])
        ]

        a_oc_base = base["rsutoc"] / base["rsdt"]
        a_oc_base[~np.isfinite(a_oc_base)] = 0.0  # this is safe
        alpha_oc_base = base["rsusoc"] / base["rsdsoc"]
        alpha_oc_base[~np.isfinite(alpha_oc_base)] = 0.0
        q_oc_down_base = base["rsdsoc"] / base["rsdt"]
        q_oc_down_base[~np.isfinite(q_oc_down_base)] = 0.0
        mu_oc_base = a_oc_base + q_oc_down_base * (1 - alpha_oc_base)
        gamma_oc_base = (mu_oc_base - q_oc_down_base) / (
            mu_oc_base - alpha_oc_base * q_oc_down_base
        )
        gamma_oc_base[~np.isfinite(gamma_oc_base)] = 0.0

        a_oc_pert = pert["rsutoc"] / pert["rsdt"]
        a_oc_pert[~np.isfinite(a_oc_pert)] = 0.0
        alpha_oc_pert = pert["rsusoc"] / pert["rsdsoc"]
        alpha_oc_pert[~np.isfinite(alpha_oc_pert)] = 0.0
        q_oc_down_pert = pert["rsdsoc"] / pert["rsdt"]
        q_oc_down_pert[~np.isfinite(q_oc_down_pert)] = 0.0
        mu_oc_pert = a_oc_pert + q_oc_down_pert * (1 - alpha_oc_pert)
        gamma_oc_pert = (mu_oc_pert - q_oc_down_pert) / (
            mu_oc_pert - alpha_oc_pert * q_oc_down_pert
        )
        gamma_oc_pert[~np.isfinite(gamma_oc_pert)] = 0.0

        a_cs_base = base["rsutcs"] / base["rsdt"]
        a_cs_base[~np.isfinite(a_cs_base)] = 0.0
        alpha_cs_base = base["rsuscs"] / base["rsdscs"]
        alpha_cs_base[~np.isfinite(alpha_cs_base)] = 0.0
        q_cs_down_base = base["rsdscs"] / base["rsdt"]
        q_cs_down_base[~np.isfinite(q_cs_down_base)] = 0.0
        mu_cs_base = a_cs_base + q_cs_down_base * (1 - alpha_cs_base)
        gamma_cs_base = (mu_cs_base - q_cs_down_base) / (
            mu_cs_base - alpha_cs_base * q_cs_down_base
        )
        gamma_cs_base[~np.isfinite(gamma_cs_base)] = 0.0

        a_cs_pert = pert["rsutcs"] / pert["rsdt"]
        a_cs_pert[~np.isfinite(a_cs_pert)] = 0.0
        alpha_cs_pert = pert["rsuscs"] / pert["rsdscs"]
        alpha_cs_pert[~np.isfinite(alpha_cs_pert)] = 0.0
        q_cs_down_pert = pert["rsdscs"] / pert["rsdt"]
        q_cs_down_pert[~np.isfinite(q_cs_down_pert)] = 0.0
        mu_cs_pert = a_cs_pert + q_cs_down_pert * (1 - alpha_cs_pert)
        gamma_cs_pert = (mu_cs_pert - q_cs_down_pert) / (
            mu_cs_pert - alpha_cs_pert * q_cs_down_pert
        )
        gamma_cs_pert[~np.isfinite(gamma_cs_pert)] = 0.0

        # Calculate cloudy values of gamma and mu
        gamma_pert = 1 - (1 - gamma_oc_pert) / (1 - gamma_cs_pert)
        mu_pert = (mu_oc_pert) / mu_cs_pert
        mu_pert[~np.isfinite(mu_pert)] = 0.0
        gamma_base = 1 - (1 - gamma_oc_base) / (1 - gamma_cs_base)
        mu_base = (mu_oc_base) / mu_cs_base
        mu_base[~np.isfinite(mu_base)] = 0.0

    aoc_base = _planetary_albedo(mu_oc_base, gamma_oc_base, alpha_oc_base)
    aoc_pert = _planetary_albedo(mu_oc_pert, gamma_oc_pert, alpha_oc_pert)
    acs_base = _planetary_albedo(mu_cs_base, gamma_cs_base, alpha_cs_base)
    acs_pert = _planetary_albedo(mu_cs_pert, gamma_cs_pert, alpha_cs_pert)

    daoc_dacld_fwd = (
        _planetary_albedo(mu_oc_base, gamma_oc_base, alpha_oc_pert) - aoc_base
    )
    daoc_dacld_bwd = aoc_pert - _planetary_albedo(
        mu_oc_pert, gamma_oc_pert, alpha_oc_base
    )

    dacs_daclr_fwd = (
        _planetary_albedo(mu_cs_base, gamma_cs_base, alpha_cs_pert) - acs_base
    )
    dacs_daclr_bwd = acs_pert - _planetary_albedo(
        mu_cs_pert, gamma_cs_pert, alpha_cs_base
    )

    dacs_dmaer_fwd = (
        _planetary_albedo(mu_cs_pert, gamma_cs_base, alpha_cs_base) - acs_base
    )
    dacs_dmaer_bwd = acs_pert - _planetary_albedo(
        mu_cs_base, gamma_cs_pert, alpha_cs_pert
    )

    dacs_dgaer_fwd = (
        _planetary_albedo(mu_cs_base, gamma_cs_pert, alpha_cs_base) - acs_base
    )
    dacs_dgaer_bwd = acs_pert - _planetary_albedo(
        mu_cs_pert, gamma_cs_base, alpha_cs_pert
    )

    # Need isolated effect of mu_cs on mu_oc holding mu_cloud fixed, and vice versa:
    new_mu_oc_pert = mu_cs_pert * mu_base  # Eq. 14
    new_mu_oc_base = mu_cs_base * mu_pert  # Eq. 14
    daoc_dmcld_fwd = (
        _planetary_albedo(new_mu_oc_base, gamma_oc_base, alpha_oc_base) - aoc_base
    )
    daoc_dmcld_bwd = aoc_pert - _planetary_albedo(
        new_mu_oc_pert, gamma_oc_pert, alpha_oc_pert
    )
    daoc_dmaer_fwd = (
        _planetary_albedo(new_mu_oc_pert, gamma_oc_base, alpha_oc_base) - aoc_base
    )
    daoc_dmaer_bwd = aoc_pert - _planetary_albedo(
        new_mu_oc_base, gamma_oc_pert, alpha_oc_pert
    )

    # Need isolated effect of gamma_cs on gamma_oc holding gamma_cloud fixed, and vice
    # versa:
    new_gamma_oc_pert = 1 - (1 - gamma_cs_pert) * (1 - gamma_base)  # Eq. 13
    new_gamma_oc_base = 1 - (1 - gamma_cs_base) * (1 - gamma_pert)  # Eq. 13
    daoc_dgcld_fwd = (
        _planetary_albedo(mu_oc_base, new_gamma_oc_base, alpha_oc_base) - aoc_base
    )
    daoc_dgcld_bwd = aoc_pert - _planetary_albedo(
        mu_oc_pert, new_gamma_oc_pert, alpha_oc_pert
    )
    daoc_dgaer_fwd = (
        _planetary_albedo(mu_oc_base, new_gamma_oc_pert, alpha_oc_base) - aoc_base
    )
    daoc_dgaer_bwd = aoc_pert - _planetary_albedo(
        mu_oc_pert, new_gamma_oc_base, alpha_oc_pert
    )

    # t1 to t9 are the coefficients of equation A2 in Zelinka et al., 2014
    forward = {}
    reverse = {}
    central = {}
    forward["t1"] = -base["rsdt"] * (1 - base["clt"]) * dacs_daclr_fwd
    forward["t2"] = -base["rsdt"] * (1 - base["clt"]) * dacs_dgaer_fwd
    forward["t3"] = -base["rsdt"] * (1 - base["clt"]) * dacs_dmaer_fwd
    forward["t4"] = -base["rsdt"] * base["clt"] * (daoc_dacld_fwd)
    forward["t5"] = -base["rsdt"] * base["clt"] * (daoc_dgaer_fwd)
    forward["t6"] = -base["rsdt"] * base["clt"] * (daoc_dmaer_fwd)
    forward["t7"] = -base["rsdt"] * base["clt"] * (daoc_dgcld_fwd)
    forward["t8"] = -base["rsdt"] * base["clt"] * (daoc_dmcld_fwd)
    forward["t9"] = -delta_clt * (rsutoc - rsutcs)
    forward["t2_clr"] = -base["rsdt"] * dacs_dgaer_fwd
    forward["t3_clr"] = -base["rsdt"] * dacs_dmaer_fwd

    # set thresholds
    # TODO: can we avoid a hard cloud fraction threshold here?
    for term in ["t4", "t5", "t6", "t7", "t8", "t9"]:
        forward[term] = np.where(
            np.logical_or(base["clt"] < cs_threshold, pert["clt"] < cs_threshold),
            0.0,
            forward[term],
        )

    # set fields to zero when incoming solar radiation is zero
    for term in ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]:
        forward[term] = np.where(
            rsdt < rsdt_threshold,
            0.0,
            forward[term],
        )

    forward["ERFariSWclr"] = forward["t2_clr"] + forward["t3_clr"]
    forward["ERFariSW"] = forward["t2"] + forward["t3"] + forward["t5"] + forward["t6"]
    forward["ERFaciSW"] = forward["t7"] + forward["t8"] + forward["t9"]
    forward["albedo"] = forward["t1"] + forward["t4"]

    reverse["t1"] = -pert["rsdt"] * (1 - pert["clt"]) * dacs_daclr_bwd
    reverse["t2"] = -pert["rsdt"] * (1 - pert["clt"]) * dacs_dgaer_bwd
    reverse["t3"] = -pert["rsdt"] * (1 - pert["clt"]) * dacs_dmaer_bwd
    reverse["t4"] = -pert["rsdt"] * pert["clt"] * (daoc_dacld_bwd)
    reverse["t5"] = -pert["rsdt"] * pert["clt"] * (daoc_dgaer_bwd)
    reverse["t6"] = -pert["rsdt"] * pert["clt"] * (daoc_dmaer_bwd)
    reverse["t7"] = -pert["rsdt"] * pert["clt"] * (daoc_dgcld_bwd)
    reverse["t8"] = -pert["rsdt"] * pert["clt"] * (daoc_dmcld_bwd)
    reverse["t9"] = -delta_clt * (rsutoc - rsutcs)
    reverse["t2_clr"] = -pert["rsdt"] * dacs_dgaer_bwd
    reverse["t3_clr"] = -pert["rsdt"] * dacs_dmaer_bwd

    # set thresholds
    for term in ["t4", "t5", "t6", "t7", "t8", "t9"]:
        reverse[term] = np.where(
            np.logical_or(base["clt"] < cs_threshold, pert["clt"] < cs_threshold),
            0.0,
            reverse[term],
        )

    # set fields to zero when incoming solar radiation is zero
    for term in ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]:
        reverse[term] = np.where(
            rsdt < rsdt_threshold,
            0.0,
            reverse[term],
        )

    reverse["ERFariSWclr"] = reverse["t2_clr"] + reverse["t3_clr"]
    reverse["ERFariSW"] = reverse["t2"] + reverse["t3"] + reverse["t5"] + reverse["t6"]
    reverse["ERFaciSW"] = reverse["t7"] + reverse["t8"] + reverse["t9"]
    reverse["albedo"] = reverse["t1"] + reverse["t4"]

    for key, _ in forward.items():
        central[key] = 0.5 * (forward[key] + reverse[key])

    if longwave:
        central["ERFariLW"], central["ERFaciLW"] = cloud_radiative_effect(base, pert)
        forward["ERFariLW"] = np.copy(central["ERFariLW"])
        reverse["ERFariLW"] = np.copy(central["ERFariLW"])
        forward["ERFaciLW"] = np.copy(central["ERFaciLW"])
        reverse["ERFaciLW"] = np.copy(central["ERFaciLW"])

    if breakdown:
        return central, forward, reverse
    return central


def create_input(
    basedir, pertdir, latout=False, longwave=False, slc=slice(0, None, None)
):
    """Extract variables from a given directory and places into dictionaries.

    It assumes that base and pert are different directories and only one
    experiment output is present in each directory.

    Slicing into time chunks is allowed and providing the filenames
    follow CMIP6 convention they should be concatenated in the correct
    order.

    Variables required are rsdt, rsus, rsds, clt, rsdscs, rsuscs, rsut, rsutcs
    An error will be raised if variables are not detected.

    Parameters
    ----------
        basedir : str
            Directory containing control climate simulation variables
        pertdir : str
            Directory containing perturbed climate simulation variables
        latout : bool, default=False
            if True, include array of latitude points in the output.
        longwave : bool, default=False
            if True, do the longwave calculation using cloud radiative effect, in
            addition to the shortwave calculation using APRP.
        slc: `slice`, optional
            Slice of indices to use from each dataset if not all of them.

    Returns
    -------
        base, pert : dict of array_like of variables needed for APRP from control
        pert: dict of variables needed for APRP from experiment
        [lat]: latitude points relating to axis 1 of arrays
    """
    base = {}
    pert = {}

    if longwave:
        varlist = [
            "rsdt",
            "rsus",
            "rsds",
            "clt",
            "rsdscs",
            "rsuscs",
            "rsut",
            "rsutcs",
            "rlut",
            "rlutcs",
        ]
    else:
        varlist = ["rsdt", "rsus", "rsds", "clt", "rsdscs", "rsuscs", "rsut", "rsutcs"]

    def _extract_files(filenames, var, directory):
        if len(filenames) == 0:
            raise RuntimeError(
                f"No variables of name {var} found in directory {directory}"
            )
        for i, filename in enumerate(filenames):
            ncfile = Dataset(filename)
            invar = ncfile.variables[var][slc, ...]
            lat = ncfile.variables["lat"][:]
            ncfile.close()
            if i == 0:
                outvar = invar
            else:
                # This works for me with CMIP6 netcdfs, but we don't have a small
                # example to test with
                outvar = np.append(outvar, invar, axis=0)  # pragma: nocover
        return outvar, lat

    for var in varlist:
        filenames = sorted(glob.glob(f"{basedir}/{var}_*.nc"))
        base[var], lat = _extract_files(filenames, var, basedir)
        filenames = sorted(glob.glob(f"{pertdir}/{var}_*.nc"))
        pert[var], lat = _extract_files(filenames, var, pertdir)

    if latout:
        return base, pert, lat
    return base, pert
