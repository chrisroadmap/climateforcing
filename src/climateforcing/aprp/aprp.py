"""APRP: the approximate partial radiative perturbation calculation."""

import copy
import glob
import warnings

import numpy as np
from netCDF4 import Dataset

# TODO: sort out refactoring for main aprp function


def _planetary_albedo(
    mu, gamma, alpha_oc, mu_cs, gamma_cs, alpha_cs, clt
):  # pylint: disable=invalid-name,too-many-arguments
    """Planetary albedo, eqs. (7, 13 & 14) of Taylor et al. (2007)."""
    # Refactor target
    mu_oc = mu_cs * mu
    gamma_oc = 1 - (1 - gamma_cs) * (1 - gamma)
    pla_cs = mu_cs * gamma_cs + (mu_cs * alpha_cs * (1 - gamma_cs) ** 2) / (
        1 - alpha_cs * gamma_cs
    )
    pla_oc = mu_oc * gamma_oc + (mu_oc * alpha_oc * (1 - gamma_oc) ** 2) / (
        1 - alpha_oc * gamma_oc
    )
    pla = (1 - clt) * pla_cs + clt * pla_oc
    #    pla[~np.isfinite(pla)] = 0
    return pla


def _calculate_overcast_terms(params):
    params["rsutoc"] = (
        params["rsut"] - (1 - params["clt"]) * (params["rsutcs"])
    ) / params["clt"]
    params["rsusoc"] = (
        params["rsus"] - (1 - params["clt"]) * (params["rsuscs"])
    ) / params["clt"]
    params["rsdsoc"] = (
        params["rsds"] - (1 - params["clt"]) * (params["rsdscs"])
    ) / params["clt"]
    for var in ["rsds", "rsus"]:
        params[f"{var}oc"] = np.where(
            params[f"{var}oc"] > params[var], np.nan, params[f"{var}oc"]
        )
    for var in ["rsds", "rsus", "rsut"]:
        params[f"{var}oc"] = np.where(params[f"{var}oc"] < 0, 0, params[f"{var}oc"])
    return params


def _calculate_parameters(input_params):
    output_params = {}

    output_params["clt"] = input_params["clt"]

    # clear sky parameters
    output_params["alpha_cs"] = (
        input_params["rsuscs"] / input_params["rsdscs"]
    )  # albedo
    q_cs = (
        input_params["rsdscs"] / input_params["rsdt"]
    )  # ratio of incident sfc flux to TOA insolation
    output_params["mu_cs"] = input_params["rsutcs"] / input_params["rsdt"] + q_cs * (
        1 - output_params["alpha_cs"]
    )  # Eq. 9
    output_params["gamma_cs"] = (output_params["mu_cs"] - q_cs) / (
        output_params["mu_cs"] - output_params["alpha_cs"] * q_cs
    )  # Eq. 10

    # overcast parameters
    output_params["alpha_oc"] = (
        input_params["rsusoc"] / input_params["rsdsoc"]
    )  # albedo
    q_oc = (
        input_params["rsdsoc"] / input_params["rsdt"]
    )  # ratio of incident sfc flux to TOA insolation
    mu_oc = input_params["rsutoc"] / input_params["rsdt"] + q_oc * (
        1 - output_params["alpha_oc"]
    )  # Eq. 9
    gamma_oc = (mu_oc - q_oc) / (mu_oc - output_params["alpha_oc"] * q_oc)  # Eq. 10

    # cloud parameters
    output_params["mu"] = (
        mu_oc / output_params["mu_cs"]
    )  # Eq. 14 sometimes this is greater than 1??
    output_params["gamma"] = (gamma_oc - 1) / (
        1 - output_params["gamma_cs"]
    ) + 1  # Eq. 13

    return output_params


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


def aprp(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements  # noqa: E501
    base, pert, longwave=False, cs_threshold=0.02, clt_percent=True, rsdt_threshold=0.1
):
    """Approximate Partial Raditive Perturbation calculation.

    This calculates the breakdown of shortwave radiative forcing into absorption
    and scattering components. When used with aerosol forcing, it can be used to
    separate the effective radiative forcing into aerosol-radiation (ERFari) and
    aerosol-cloud (ERFaci) components.

    Parameters
    ----------
        base, pert : dict of np.ndarray
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
        cs_threshold : float, default=0.02
            minimum cloud fraction (0-1 scale) for calculation of cloudy-sky APRP. If
            either perturbed or control run cloud fraction is below this, set the APRP
            flux to zero. It is recommended to use a small positive value, as the
            cloud fraction appears in the denominator of the calculation. Taken from
            Mark Zelinka's implementation.
        clt_percent : bool, default=True
            is cloud fraction from base and pert in percent (True) or 0-1 scale
            (False)
        rsdt_threshold : float, default=0.1
            Set TOA downwelling radiation to zero below a certain threshold (W m**-2).
            A small positive value here brings the sum of APRP components slightly
            closer to the net forcing.

    Returns
    -------
        results : dict of np.ndarray
            Components of APRP as defined by equation A2 of [1]_:
                "albedo"
                "noncloud_scat"
                "noncloud_abs"
                "cloud_scat"
                "cloud_abs"
                "cloud_amount"
                "ERFariSW" = "noncloud_scat" + "noncloud_abs"
                "ERFaciSW" = "cloud_scat" + "cloud_abs" + "cloud_amount"
            if `longwave`=`True`, the following are also calculated from the LW fluxes:
                "ERFariLW"
                "ERFaciLW"

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
        base = _calculate_overcast_terms(base)
        pert = _calculate_overcast_terms(pert)

    base_params = _calculate_parameters(base)
    pert_params = _calculate_parameters(pert)

    base_albedo = _planetary_albedo(**base_params)
    pert_albedo = _planetary_albedo(**pert_params)

    # forward
    forward = {}
    for subs in base_params:
        in_params = copy.deepcopy(base_params)
        in_params[subs] = pert_params[subs]
        forward[subs] = _planetary_albedo(**in_params) - base_albedo

    # backward, and central as mean of forward and backward
    reverse = {}
    central = {}
    for subs in pert_params:
        in_params = copy.deepcopy(pert_params)
        in_params[subs] = base_params[subs]
        reverse[subs] = pert_albedo - _planetary_albedo(**in_params)
        central[subs] = 0.5 * (forward[subs] + reverse[subs])

    for var in ["clt", "alpha_oc", "mu", "gamma"]:
        central[var] = np.where(
            np.logical_or(base["clt"] < cs_threshold, pert["clt"] < cs_threshold),
            0.0,
            central[var],
        )

    rsdt = 0.5 * (base["rsdt"] + pert["rsdt"])

    results = {}
    results["albedo"] = -(central["alpha_cs"] + central["alpha_oc"]) * rsdt
    results["ERFari_scat"] = -central["gamma_cs"] * rsdt
    results["ERFari_abs"] = -central["mu_cs"] * rsdt
    results["ERFaci_scat"] = -central["gamma"] * rsdt
    results["ERFaci_abs"] = -central["mu"] * rsdt
    results["ERFaci_amt"] = -central["clt"] * rsdt

    for var in [
        "albedo",
        "ERFari_scat",
        "ERFari_abs",
        "ERFaci_scat",
        "ERFaci_abs",
        "ERFaci_amt",
    ]:
        results[var] = np.where(rsdt < rsdt_threshold, 0, results[var])

    results["ERFariSW"] = results["ERFari_scat"] + results["ERFari_abs"]
    results["ERFaciSW"] = (
        results["ERFaci_scat"] + results["ERFaci_abs"] + results["ERFaci_amt"]
    )

    if longwave:
        results["ERFariLW"], results["ERFaciLW"] = cloud_radiative_effect(base, pert)

    return results


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
