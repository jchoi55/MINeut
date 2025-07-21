import numpy as np
import vector
import vegas as vg
from collections import OrderedDict
from collections import defaultdict
from functools import partial
from scipy.special import spence

from mineut import const


def gauss_pdf(x, x0, sigma):
    if sigma == 0:
        return np.ones_like(x)
    else:
        return np.exp(-((x - x0) ** 2) / (2 * sigma**2)) / sigma / np.sqrt(2 * np.pi)


def Fnue0(x):
    return 6 * x**2 * (1 - x)


def Fnumu0(x):
    return x**2 * (3 - 2 * x)


def Jnue0(x):
    return 6 * x**2 * (1 - x)


def Jnumu0(x):
    return x**2 * (1 - 2 * x)


def L_Spence(x):
    """
    L(x) = - int_0^x log(1 - t)/t dt
    """
    return spence(1 - x)


def k_radiative(x):
    return 2 * L_Spence(x) + 2 * np.pi**2 / 3 + np.log(1 - x) ** 2


def Fnumu1(x):
    return (
        Fnumu0(x) * k_radiative(x)
        + 1 / 6 * (41 - 36 * x + 42 * x**2 - 16 * x**3) * np.log(1 - x)
        + 1 / 12 * x * (82 - 153 * x + 86 * x**2)
    )


def Jnumu1(x):
    term1 = Jnumu0(x) * k_radiative(x)
    term2 = (1 / 6) * (11 - 36 * x + 14 * x**2 - 16 * x**3 - 4 / x) * np.log(1 - x)
    term3 = (1 / 12) * (-8 + 18 * x - 103 * x**2 + 78 * x**3)
    return term1 + term2 + term3


def Fnue1(x):
    term1 = Fnue0(x) * k_radiative(x)
    term2 = (1 - x) * (
        (5 + 8 * x + 8 * x**2) * np.log(1 - x) + (1 / 2) * x * (10 - 19 * x)
    )
    return term1 + term2


def Jnue1(x):
    term1 = Jnue0(x) * k_radiative(x)
    term2 = (1 - x) * (
        (-3 + 12 * x + 8 * x**2 + 4 / x) * np.log(1 - x)
        + (1 / 2) * (8 - 2 * x - 15 * x**2)
    )
    return term1 + term2


def mudecay_matrix_element_sqr(
    x_nu, costheta, muon_polarization, muon_charge, nuflavor, NLO=True
):
    if "nue" in nuflavor:
        return (
            Fnue0(x_nu)
            - muon_charge * muon_polarization * Jnue0(x_nu) * costheta
            - NLO
            * const.alphaQED
            / 2
            / np.pi
            * (Fnue1(x_nu) - Jnue1(x_nu) * muon_charge * muon_polarization * costheta)
        )

    elif "numu" in nuflavor:
        return (
            Fnumu0(x_nu)
            - muon_charge * muon_polarization * Jnumu0(x_nu) * costheta
            - NLO
            * const.alphaQED
            / 2
            / np.pi
            * (Fnumu1(x_nu) - Jnumu1(x_nu) * muon_charge * muon_polarization * costheta)
        )
    else:
        raise ValueError(f"nuflavor {nuflavor} not recognized.")


class NLOmudecay_pol(vg.BatchIntegrand):
    def __init__(self, dim, MC_case):
        """
        Initialize the NLOmudecay_pol class.

        Args:
            dim (int): The dimension of the integrand.
            MC_case (DarkNews.MC.MC_events): The main Monte-Carlo class of DarkNews.
        """

        self.dim = dim
        self.MC_case = MC_case

        # Find the normalization factor
        self.norm = {}
        self.norm["diff_decay_rate"] = 1
        # normalize integrand with an initial throw
        _throw = self.__call__(
            np.random.rand(self.dim, 2000), np.ones((self.dim, 2000))
        )
        for key, val in _throw.items():
            self.norm[key] = np.mean(val)
            # cannot normalize zero integrand
            if self.norm[key] == 0.0:
                print(f"Warning: mean of integrand {key} is zero. Vegas may break.")
                self.norm[key] = 1

    def __call__(self, x, jac):

        MC_case = self.MC_case
        ######################################
        # mu -> e nu nu
        i_var = 0  # 0 is for the momentum pmu
        costheta = (2.0) * x[:, i_var] - 1.0

        i_var += 1  # 2 is for x_nu = 2 Enu/ m_mu
        x_nu = x[:, i_var]  # 0 to 1

        dgamma = mudecay_matrix_element_sqr(
            x_nu=x_nu,
            costheta=costheta,
            muon_polarization=MC_case.muon_polarization,
            muon_charge=MC_case.muon_charge,
            nuflavor=MC_case.nuflavor,
            NLO=MC_case.NLO,
        )

        # hypercube jacobian (vegas hypercube --> physical limits) transformation
        dgamma *= const.Gf**2 * const.m_mu**5 / 192 / np.pi**3

        dgamma *= 2  # d costheta
        dgamma *= 1  # d x_nu

        ##############################################
        # return all differential quantities of interest
        self.int_dic = OrderedDict()
        self.int_dic["diff_decay_rate"] = dgamma

        ##############################################
        # normalization
        self.int_dic["diff_decay_rate"] /= self.norm["diff_decay_rate"]

        return self.int_dic


# Three body decay
def three_body_decay_x_costheta(
    samples, boost=False, m1=1, m2=0, m3=0, m4=0, rng=np.random.random
):

    if not samples:
        raise ValueError("Error! No samples were passed to three_body_decay.")
    else:
        # get sample size of the first item
        sample_size = np.shape(list(samples.values())[0])[0]

    # Mandelstam t = m23^2

    x_nu = samples["x_nu"]
    costheta = samples["costheta"]

    M = m1
    Enu = M / 2 * x_nu

    # p1
    PmuCM = vector.array(
        {
            "E": np.full(sample_size, m1),
            "px": np.zeros(sample_size),
            "py": np.zeros(sample_size),
            "pz": np.zeros(sample_size),
        }
    )

    # p2
    PnuCM = vector.array(
        {
            "E": Enu,
            "pt": Enu * np.sqrt(1 - costheta**2),
            "pz": Enu * costheta,
            "phi": np.random.uniform(0, 2 * np.pi, size=sample_size),
        }
    )

    # four-momenta in the LAB frame
    if boost:
        pBoost = np.sqrt(boost["EP_LAB"] ** 2 - m1**2)
        p4Boost = vector.array(
            {
                "E": boost["EP_LAB"],
                "pt": pBoost * np.sqrt(1 - boost["costP_LAB"] ** 2),
                "pz": pBoost * boost["costP_LAB"],
                "phi": boost["phiP_LAB"],
            }
        )

        # Outgoing neutrino
        PmuLAB_decay = PmuCM.boost_p4(p4Boost)
        # Outgoing neutrino
        PnuLAB_decay = PnuCM.boost_p4(p4Boost)

        return PmuLAB_decay, PnuLAB_decay

    # four-momenta in the parent particle rest frame
    else:
        return PmuCM, PnuCM


class GeneratorEngine(object):
    def __init__(
        self,
        mudecay_model="NLOmudecay_pol",
        Mparent=const.m_mu,
        Mdaughter=const.m_e,
        nuflavor="mu",
        NLO=True,
        muon_polarization=-1,
        muon_charge=+1,
        mnu1=0,
        mnu2=0,
        NINT=10,
        NINT_warmup=10,
        NEVAL=1e5,
        NEVAL_warmup=1e4,
        save_mem=True,
    ):
        """
        Initialize the GeneratorEngine class.

        This class is responsible for generating muon decay events using the vegas.

        All decays are generated in the muon rest frame, so the parent particle is always at rest.

        Later, the lattice functions will boost all events according to the beam properties.

        Args:
            mudecay_model (str, optional): Muon decay model and amplitude to be used. Defaults to "NLOmudecay_pol".
            Mparent (_type_, optional): Mass of the parent (muon). Defaults to const.m_mu.
            Mdaughter (_type_, optional): Mass of the daughter (electron). Defaults to const.m_e.
            nuflavor (str, optional): Flavor of the neutrino. Defaults to "mu".
            NLO (bool, optional): Whether to use NLO radiative corrections. Defaults to True.
            muon_polarization (int, optional): Polarization of the muon. Defaults to -1.
            muon_charge (int, optional): Charge of the muon. Defaults to +1.
            mnu1 (int, optional): Mass of the first neutrino. Defaults to 0.
            mnu2 (int, optional): Mass of the second neutrino. Defaults to 0.
            NINT (int, optional): Number of integration points. Defaults to 10.
            NINT_warmup (int, optional): Number of warmup integration points. Defaults to 10.
            NEVAL (int, optional): Number of evaluation points. Defaults to 1e5.
            NEVAL_warmup (int, optional): Number of warmup evaluation points. Defaults to 1e4.
            save_mem (bool, optional): Whether to save memory. Defaults to True.
        """

        # set target properties
        self.Mparent = Mparent
        self.Mdaughter = Mdaughter
        self.mnu1 = mnu1
        self.mnu2 = mnu2
        self.NLO = NLO
        self.muon_polarization = muon_polarization
        self.nuflavor = nuflavor
        self.muon_charge = muon_charge
        self.mudecay_model = mudecay_model

        self.NINT = NINT
        self.NINT_warmup = NINT_warmup
        self.NEVAL = NEVAL
        self.NEVAL_warmup = NEVAL_warmup

        self.save_mem = save_mem

    def run_vegas(
        self,
        savestr=None,
        **kwargs,
    ):
        """
        Function that calls vegas evaluations.
        This function defines the vegas parameters used in event generation.

        Returns:
            integ (vegas.Integrator): with the evaluated integrals.
        """

        # warm up the MC, adapting to the integrand
        self.integ(
            self.batch_f,
            nitn=self.NINT_warmup,
            neval=self.NEVAL_warmup,
            uses_jac=True,
            **kwargs,
        )
        # sample again, now saving result and turning off further adaption
        return self.integ(
            self.batch_f,
            nitn=self.NINT,
            neval=self.NEVAL,
            uses_jac=True,
            saveall=savestr,
            **kwargs,
        )

    def get_vegas_samples(self, return_jac=False):
        """_summary_

        Args:
            return_jac (bool, optional): if True, returns the jacobian of the integrand as well. Defaults to False.

        Raises:
            ValueError: if the integrand evaluates to nan

        Returns:
            tuple of np.ndarrays:
        """

        unit_samples = self.batch_f.dim * [[]]
        weights = defaultdict(partial(np.ndarray, 0))

        for x, y, wgt in self.integ.random_batch(yield_y=True):
            # compute integrand on samples including jacobian factors
            if self.integ.uses_jac:
                jac = self.integ.map.jac1d(y)
            else:
                jac = None

            fx = self.batch_f(x, jac=jac)
            # weights
            for fx_i in fx.keys():
                if np.any(np.isnan(fx[fx_i])):
                    raise ValueError(f"integrand {fx_i} evaluates to nan")
                weights[fx_i] = np.append(weights[fx_i], wgt * fx[fx_i])

            # MC samples in unit hypercube
            for i in range(self.batch_f.dim):
                unit_samples[i] = np.append(unit_samples[i], x[:, i])

        if return_jac:
            return np.array(unit_samples), weights, jac
        else:
            return np.array(unit_samples), weights

    def build_4momenta(self, vsamples):
        """
        Construct the four momenta of all particles in the upscattering+decay process from the
        vegas weights.

        Args:
                vsamples (np.ndarray): integration samples obtained from vegas
                                as hypercube coordinates. Always in the interval [0,1].

        Returns:
                dict: each key corresponds to a set of four momenta for a given particle involved,
                        so the values are 2D np.ndarrays with each row a different event and each column a different
                        four momentum component. Contains also the weights.
        """

        four_momenta = {}

        ########################
        # decay
        # energy of projectile
        # pP = (self.pmax - self.pmin) * np.array(vsamples[0]) + self.pmin
        # EP = np.sqrt(self.Mparent**2 + pP**2)
        masses_decay = {
            "m1": self.Mparent,
            "m2": self.Mdaughter,
            "m3": self.mnu1,
            "m4": self.mnu2,
        }

        # # parent particle boost parameters
        # boost_parent = {
        #     "EP_LAB": EP,
        #     "costP_LAB": np.ones((np.size(EP),)),
        #     "phiP_LAB": np.zeros((np.size(EP),)),
        # }

        decay_samples = {"costheta": 2 * vsamples[0] - 1, "x_nu": vsamples[1]}

        # Pmu Pnu(bar)
        (
            PmuLAB_decay,
            PnuLAB_decay,
        ) = three_body_decay_x_costheta(decay_samples, boost=None, **masses_decay)

        four_momenta["P_decay_mu"] = PmuLAB_decay
        four_momenta["P_decay_nu"] = PnuLAB_decay

        return four_momenta

    def get_events(self):

        if self.mudecay_model == "NLOmudecay_pol":
            DIM = 2  # dim of phase space
            self.batch_f = NLOmudecay_pol(dim=DIM, MC_case=self)
            self.integ = vg.Integrator(DIM * [[0.0, 1.0]])

            _ = self.run_vegas()
        else:
            raise ValueError(f"Model {self.mudecay_model} not recognized.")

        #########################
        # Get the integration variables and weights used by vegas
        samples, weights = self.get_vegas_samples(return_jac=False)

        # Build the four-momenta of the particles in the decay using the vegas samples
        events_dict = self.build_4momenta(vsamples=samples)

        # Save mu-decay rest frame variables for easier reweighting later
        events_dict["costheta_CM"] = 2 * samples[0] - 1
        events_dict["x_CM"] = samples[1]

        # Include event weights
        events_dict["w_flux"] = (
            weights["diff_decay_rate"] * self.batch_f.norm["diff_decay_rate"]
        )

        if self.save_mem:
            del weights
            del samples

        return events_dict
