import numpy as np
from scipy.interpolate import interp1d

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from DarkNews import const

nuf_dict = {
    "nue": "e",
    "nuebar": "e",
    "numu": "m",
    "numubar": "m",
    "nutau": "t",
    "nutaubar": "t",
}


# """Returns necessary cross neutrino cross sections from xsecs"""
# log10E, sigmaeo, sigmamuo, _, sigmaebaro, sigmamubaro, _ = np.genfromtxt(
#     files("MuC.xsec_data").joinpath("XCC.dat").open(), unpack=True
# )
# exs = 10 ** (log10E)
# exs = np.append(exs, 1e4)

# # adding values linearily up to 10 TeV
# sigmae = np.append(sigmaeo, 7.046434082801659e-1)
# sigmamu = np.append(sigmamuo, 7.046434082801659e-1)
# sigmaebar = np.append(sigmaebaro, 3.758631195493489e-1)
# sigmamubar = np.append(sigmamubaro, 3.758631195493489e-1)

# # generating interpolators
# total_sigmanue = interp1d(exs, sigmae * exs * 1e-38, bounds_error=False, fill_value=0.0)
# total_sigmanuebar = interp1d(
#     exs, sigmaebar * exs * 1e-38, bounds_error=False, fill_value=0.0
# )
# total_sigmanumu = interp1d(
#     exs, sigmamu * exs * 1e-38, bounds_error=False, fill_value=0.0
# )
# total_sigmanumubar = interp1d(
#     exs, sigmamubar * exs * 1e-38, bounds_error=False, fill_value=0.0
# )


""" 
    Neutrino-electron elastic scattering functions 
     
     * These are all calculable
     * NOTE: neglecting radiative corrections of O(alpha_EM) for now 
"""


def nue_elastic_sigma(Enu, nuf, s2w=0.2334, m=const.m_e, Te_min=0, Te_max=np.inf):
    """
    Calculate the total cross section ignoring terms proportional to alpha_EM.

    Parameters:
    E_nu : float
        Neutrino energy.
    C_LL : float
        Coupling constant for LL component.
    C_LR : float
        Coupling constant for LR component.
    m : float
        Mass of the lepton in GeV.
    Te_min : float
        Minimum kinetic energy of the outgoing electron.

    Returns:
    tuple: Total cross sections for (neutrino-electron -> neutrino-lepton-electron) and (antineutrino-electron -> antineutrino-lepton-electron).
    """

    # Neutrino-electron scattering tools
    C_LL_dic = {}
    C_LL_dic["e"] = 1 + s2w - 0.5  # 0.7276
    C_LL_dic["m"] = s2w - 0.5  # -0.2730
    C_LL_dic["t"] = s2w - 0.5  # -0.2730
    C_LL = C_LL_dic[nuf_dict[nuf]]
    C_LR = s2w

    s = 2 * m * Enu  # + m**2
    prefactor = const.Gf**2 * s / np.pi

    y_max = np.minimum(2 * Enu / (2 * Enu + m), Te_max / Enu)
    y_min = Te_min / Enu
    y_range = y_max - y_min

    if "bar" in nuf:
        term1 = (C_LR**2) * y_range
        term2 = (C_LL**2) * (
            y_range - (y_max**2 - y_min**2) + (y_max**3 - y_min**3) / 3
        )
        term3 = (-C_LL * C_LR * m / (2 * Enu)) * y_range**2
        sigma = prefactor * (term1 + term2 + term3)

    else:
        term1 = (C_LL**2) * y_range
        term2 = (C_LR**2) * (
            y_range - (y_max**2 - y_min**2) + (y_max**3 - y_min**3) / 3
        )
        term3 = (-C_LL * C_LR * m / (2 * Enu)) * (y_max**2 - y_min**2)

        sigma = prefactor * (term1 + term2 + term3)
    return sigma * const.invGeV2_to_cm2


def nue_elastic_dsigma_dy(y, Enu, nuf, s2w=0.2334, m=const.m_e):
    """
    Calculate the differential cross section ignoring terms proportional to alpha_EM.

    Parameters:
    C_LL : float
        Coupling constant for LL component.
    C_LR : float
        Coupling constant for LR component.
    E_nu : float
        Neutrino energy.
    y : float
        Fractional energy transfer.
    m : float
        Mass of the lepton.

    Returns:
    tuple: Differential cross sections for (neutrino-electron -> neutrino-lepton-electron) and (antineutrino-electron -> antineutrino-lepton-electron).
    """
    C_LL_dic = {}
    C_LL_dic["e"] = 1 + s2w - 0.5  # 0.7276
    C_LL_dic["m"] = s2w - 0.5  # -0.2730
    C_LL_dic["t"] = s2w - 0.5  # -0.2730
    C_LL = C_LL_dic[nuf_dict[nuf]]
    C_LR = s2w
    s = 2 * m * Enu + m**2

    if "bar" in nuf:
        # Differential cross section for antineutrino scattering
        dsigma_dy = (const.Gf**2 * s / np.pi) * (
            (C_LR**2) + (C_LL**2) * (1 - y) ** 2 - (C_LL * C_LR * m * y) / Enu
        )
    else:
        # Differential cross section for neutrino scattering
        dsigma_dy = (const.Gf**2 * s / np.pi) * (
            (C_LL**2) + (C_LR**2) * (1 - y) ** 2 - (C_LL * C_LR * m * y) / Enu
        )
    return np.where((y >= 0) & (y <= 1), dsigma_dy, 0) * const.invGeV2_to_cm2


def inverse_lepton_decay_sigma(Enu, nui, lepton):
    """
    Calculate the differential cross section ignoring terms proportional to alpha_EM.

    Returns:
    tuple: Differential cross sections for (neutrino-electron -> neutrino-lepton-electron) and (antineutrino-electron -> antineutrino-lepton-electron).
    """
    if lepton == "m":
        mlepton = const.m_mu
    elif lepton == "t":
        mlepton = const.m_tau
    else:
        return None

    if nui not in ["numu", "nutau", "nuebar"]:
        return Enu * 0

    s = 2 * Enu * const.m_e + const.m_e**2
    kinematically_allowed = s > mlepton**2

    ymax = 1 - mlepton**2 / (s)

    prefactor = 2 * const.m_e * const.Gf**2 * Enu / np.pi

    if nui != "nuebar":
        # Differential cross section for neutrino scattering
        sigma = (
            prefactor * (1 - (mlepton**2 - const.m_e**2) / (s - const.m_e**2)) * ymax
        )
    else:
        # Differential cross section for antineutrino scattering
        sigma = prefactor * (
            ymax
            - ((const.m_e**2 - mlepton**2) * (ymax - 2) * ymax) / (4 * Enu * const.m_e)
            + (ymax / 3 - 1) * ymax**2
        )

    return np.where(kinematically_allowed, sigma, 0) * const.invGeV2_to_cm2


"""
 Cross sections taken from Alfonso's paper

 * we will eventually assume isoscalar targets, so averaging p and n
 * everything is in cm^2
 * coherent channels are calculated for C or O, and then divided by number of total nucleons 
    (we approximate the xsec on other targets by ratios of Z_C^2/Z_X^2, with C carbon and X another nucleus).
"""

kwargs_interp = {"kind": "linear", "fill_value": 0, "bounds_error": False}

data_p = np.genfromtxt(
    files("MuC.xsec_data").joinpath("Alfonso_data_proton.dat").open()
).T
data_n = np.genfromtxt(
    files("MuC.xsec_data").joinpath("Alfonso_data_neutron.dat").open()
).T
Enu_data = data_p[0]

sigma_lightquark_CC_nu = interp1d(
    Enu_data, (data_n[1] + data_p[1]) / 2 * 1e-36, **kwargs_interp
)
sigma_lightquark_CC_nubar = interp1d(
    data_p[0], (data_n[6] + data_p[6]) / 2 * 1e-36, **kwargs_interp
)
sigma_charm_CC_nu = interp1d(
    Enu_data,
    (data_n[1] * data_n[3] + data_p[1] * data_p[3]) / 2 * 1e-36,
    **kwargs_interp,
)
sigma_charm_CC_nubar = interp1d(
    Enu_data,
    (data_n[6] * data_n[8] + data_p[6] * data_p[8]) / 2 * 1e-36,
    **kwargs_interp,
)

sigma_bottom_CC_nu = interp1d(
    Enu_data,
    (data_n[1] * data_n[4] + data_p[1] * data_p[4]) / 2 * 1e-36,
    **kwargs_interp,
)
sigma_bottom_CC_nubar = interp1d(
    Enu_data,
    (data_n[6] * data_n[9] + data_p[6] * data_p[9]) / 2 * 1e-36,
    **kwargs_interp,
)

# NOTE: Top quark threshold > 10 TeV
# sigma_top_CC_nu = interp1d(
#     Enu_data,
#     (data_n[1] * data_n[3] + data_p[1] * data_p[3]) / 2 * 1e-36,
#     **kwargs_interp,
# )
# sigma_top_CC_nubar = interp1d(
#     Enu_data,
#     (data_n[6] * data_n[8] + data_p[6] * data_p[8]) / 2 * 1e-36,
#     **kwargs_interp,
# )


# Neutral current -- NOTE: approximate (assumes isoscalar targets)
def sigma_NC_nu(Enu):
    return 0.335 * sigma_lightquark_CC_nu(Enu)


def sigma_NC_nubar(Enu):
    return 0.335 * sigma_lightquark_CC_nubar(Enu)


# Quasi-Elastic scattering -- NOTE: approximate (also assumes isoscalar targets)
def sigma_QE_nu(Enu):
    return 0.9 * 1e-38 * np.ones(len(Enu)) / 2


def sigma_QE_nubar(Enu):
    return 0.9 * 1e-38 * np.ones(len(Enu)) / 2


# Coherent pion production (calculation on Carbon)
e, ratio = np.genfromtxt(
    files("MuC.xsec_data").joinpath("tot_xsec_cohpi0_C.dat").open(), unpack=True
)
sigma_cohpi0 = interp1d(
    e,
    ratio * 1e-40 / 12,
    kind="linear",
    fill_value=ratio[-1] * 1e-40 / 12,
    bounds_error=False,
)

# Resoannt rho- production (on nuebar on electrons)
e, ratio = np.genfromtxt(
    files("MuC.xsec_data").joinpath("R_weak_rho.dat").open(), unpack=True
)
sigma_resonant_rho = interp1d(
    e,
    ratio * inverse_lepton_decay_sigma(e, nui="nuebar", lepton="m"),
    kind="linear",
    fill_value=0,
    bounds_error=False,
)

# Resoannt K*- production (on nuebar on electrons)
e, ratio = np.genfromtxt(
    files("MuC.xsec_data").joinpath("R_weak_Kstar.dat").open(), unpack=True
)
sigma_resonant_Kstar = interp1d(
    e,
    ratio * inverse_lepton_decay_sigma(e, nui="nuebar", lepton="m"),
    kind="linear",
    fill_value=0,
    bounds_error=False,
)


# Total cross sections -- neglecting nu-e and W production
# Assume (sigma_e = sigma_mu)
total_xsecs = {}
total_xsecs["nue"] = (
    lambda x: sigma_lightquark_CC_nu(x)
    + sigma_charm_CC_nu(x)
    + sigma_bottom_CC_nu(x)
    + sigma_NC_nu(x)
)
total_xsecs["nuebar"] = (
    lambda x: sigma_lightquark_CC_nubar(x)
    + sigma_charm_CC_nubar(x)
    + sigma_bottom_CC_nubar(x)
    + sigma_NC_nubar(x)
)
total_xsecs["numu"] = (
    lambda x: sigma_lightquark_CC_nu(x)
    + sigma_charm_CC_nu(x)
    + sigma_bottom_CC_nu(x)
    + sigma_NC_nu(x)
)
total_xsecs["numubar"] = (
    lambda x: sigma_lightquark_CC_nubar(x)
    + sigma_charm_CC_nubar(x)
    + sigma_bottom_CC_nubar(x)
    + sigma_NC_nubar(x)
)


"""
    Neutrino-nucleus trident scattering functions

     * Calculated by B. Zhou and J. Beacom
"""


def get_trident_xsec(nui, nuf, minus_lep, plus_lep, target):
    """
    Get the trident cross section for a given channel.

    Parameters:
    nui : str
        Initial neutrino flavor.
    nuf : str
        Final neutrino flavor.
    minus_lep : str
        Negative charged lepton flavor.
    plus_lep : str
        Positive lepton flavor.
    target : str
        Target nucleus.

    NOTE: neutrino and antineutrino channels (CP conjugates) have the same xsec
    E.g.,
        CP (nui --> nuf + minus_lep + plus_lep) = nuibar --> nufbar + plus_lep + minus_lep

    Returns:
    tuple: Energy and cross section for the given channel.
    """
    try:
        xsec = np.loadtxt(
            files("MuC.xsec_data.trident_production")
            .joinpath(
                f"v{nui}{target}TOv{nuf}{minus_lep}{plus_lep.capitalize()}X_tot.dat"
            )
            .open()
        ).T
    except FileNotFoundError:
        try:
            xsec = np.loadtxt(
                files("MuC.xsec_data.trident_production")
                .joinpath(
                    f"v{nui}{target}TO{minus_lep}v{nuf}{plus_lep.capitalize()}X_tot.dat"
                )
                .open()
            ).T
        except FileNotFoundError:
            # print(
            # f"Could not find trident xsec for v{nui}{target}TO{minus_lep}v{nuf}{plus_lep.capitalize()}X"
            # )
            return lambda x: 0

    return interp1d(xsec[0], xsec[1], **kwargs_interp)


def get_cross_sections(Enu, nui, Z, A):

    xsecs = {}

    # Neutrino-nucleon cross sections (NOTE: Assume isoscalar targets)
    if "bar" in nui:
        xsecs[f"{nui}_CC_light"] = sigma_lightquark_CC_nubar(Enu)
        xsecs[f"{nui}_CC_charm"] = sigma_charm_CC_nubar(Enu)
        xsecs[f"{nui}_CC_bottom"] = sigma_bottom_CC_nubar(Enu)
        xsecs[f"{nui}_NC"] = sigma_NC_nubar(Enu)
        xsecs[f"{nui}_QE"] = sigma_QE_nu(Enu)
    else:
        xsecs[f"{nui}_CC_light"] = sigma_lightquark_CC_nu(Enu)
        xsecs[f"{nui}_CC_charm"] = sigma_charm_CC_nu(Enu)
        xsecs[f"{nui}_CC_bottom"] = sigma_bottom_CC_nu(Enu)
        xsecs[f"{nui}_NC"] = sigma_NC_nu(Enu)
        xsecs[f"{nui}_QE"] = sigma_QE_nubar(Enu)

    # Elastic neutrino-electron scattering
    xsecs[f"{nui}_ES_e"] = nue_elastic_sigma(Enu, nui) * Z / A

    # Inverse lepton decay
    xsecs[f"{nui}_invdecay_mu"] = (
        inverse_lepton_decay_sigma(Enu, nui=nui, lepton="m") * Z / A
    )
    xsecs[f"{nui}_invdecay_tau"] = (
        inverse_lepton_decay_sigma(Enu, nui=nui, lepton="t") * Z / A
    )

    # if nui == "nuebar":
    # xsecs[f"{nui}_invdecay_mu"] = inverse_lepton_decay_sigma(Enu, nui=nui, lepton="m")
    # xsecs[f"{nui}_invdecay_tau"] = inverse_lepton_decay_sigma(Enu, nui=nui, lepton="t")

    # Coherent pion production
    xsecs[f"{nui}_pi0_coh"] = sigma_cohpi0(Enu) * (
        (Z / 6) ** 2 / (A)
    )  # Approximation for other targets (Z=6 for Carbon)

    # resonant rho- production
    xsecs[f"{nui}_resonant_rho-"] = (
        sigma_resonant_rho(Enu) * (nui == "nuebar") * (Z / A)
    )
    xsecs[f"{nui}_resonant_Kstar-"] = (
        sigma_resonant_Kstar(Enu) * (nui == "nuebar") * (Z / A)
    )

    flavors = ["e", "m", "l"]
    for nuf in flavors:
        for minus_lep in flavors:
            for plus_lep in flavors:
                tri_xsec = get_trident_xsec(
                    nuf_dict[nui], nuf, minus_lep, plus_lep, "O16"
                )
                if tri_xsec is not None:
                    xsecs[f"{nui}_{nuf}{minus_lep}{plus_lep}_tri"] = tri_xsec(Enu) * (
                        (Z / 8) ** 2 / A
                    )  # Approximation for other targets (Z=8 for Oxygen)

    xsecs[f"{nui}_total"] = total_xsecs[nui](Enu)
    return xsecs
