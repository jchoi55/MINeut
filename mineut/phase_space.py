import numpy as np
import vector

from mineut.const import rng_interval


########################################################
# Kinematical limits
# 1 --> 3 decays (decay mandelstam)
def three_body_umax(m1, m2, m3, m4, t):
    return 1 / 4 * (
        ((m1) ** (2) + ((m2) ** (2) + (-1 * (m3) ** (2) + -1 * (m4) ** (2))))
    ) ** (2) * (t) ** (-1) + -1 * (
        (
            (
                (
                    -1 * (m2) ** (2)
                    + 1
                    / 4
                    * (t) ** (-1)
                    * (((m2) ** (2) + (-1 * (m3) ** (2) + t))) ** (2)
                )
            )
            ** (1 / 2)
            + -1
            * (
                (
                    -1 * (m4) ** (2)
                    + 1
                    / 4
                    * (t) ** (-1)
                    * ((-1 * (m1) ** (2) + ((m4) ** (2) + t))) ** (2)
                )
            )
            ** (1 / 2)
        )
    ) ** (
        2
    )


def three_body_umin(m1, m2, m3, m4, t):
    return 1 / 4 * (
        ((m1) ** (2) + ((m2) ** (2) + (-1 * (m3) ** (2) + -1 * (m4) ** (2))))
    ) ** (2) * (t) ** (-1) + -1 * (
        (
            (
                (
                    -1 * (m2) ** (2)
                    + 1
                    / 4
                    * (t) ** (-1)
                    * (((m2) ** (2) + (-1 * (m3) ** (2) + t))) ** (2)
                )
            )
            ** (1 / 2)
            + (
                (
                    -1 * (m4) ** (2)
                    + 1
                    / 4
                    * (t) ** (-1)
                    * ((-1 * (m1) ** (2) + ((m4) ** (2) + t))) ** (2)
                )
            )
            ** (1 / 2)
        )
    ) ** (
        2
    )


def three_body_tmax(m1, m2, m3, m4):
    return (m1 - m4) ** 2


def three_body_tmin(m1, m2, m3, m4):
    return (m2 + m3) ** 2


# Two body decay
# p1 (k1) --> p2(k2) p3(k3)
def two_body_decay(samples, boost=False, m1=1, m2=0, m3=0, rng=np.random.random):

    if not samples:
        sample_size = np.shape(list(boost.values())[0])[0]
    else:
        # get sample size of the first item
        sample_size = np.shape(list(samples.values())[0])[0]

    # cosine of the angle between k2 and z axis
    if "unit_cost" in samples.keys():
        cost = 2 * samples["unit_cost"] - 1
    elif "cost" in samples.keys():
        cost = samples["cost"]
    else:
        cost = rng_interval(sample_size, -1, 1, rng=rng)

    E1CM_decay = np.full_like(cost, m1)
    E2CM_decay = np.full_like(cost, (m1**2 + m2**2 - m3**2) / 2.0 / m1)
    E3CM_decay = np.full_like(cost, (m1**2 - m2**2 + m3**2) / 2.0 / m1)

    p2CM_decay = np.full_like(cost, np.sqrt(E2CM_decay**2 - m2**2))
    # p3CM_decay = np.full_like(cost, np.sqrt(E3CM_decay**2 - m3**2))

    # azimuthal angle of k2
    if "unit_phiz" in samples.keys():
        phiz = 2 * samples["unit_phiz"] - 1
    elif "phiz" in samples.keys():
        phiz = samples["phiz"]
    else:
        phiz = rng_interval(sample_size, 0.0, 2 * np.pi, rng=rng)

    P1CM_decay = vector.array(
        {
            "E": E1CM_decay,
            "px": np.zeros(sample_size),
            "py": np.zeros(sample_size),
            "pz": np.zeros(sample_size),
        }
    )
    P2CM_decay = vector.array(
        {
            "E": E2CM_decay,
            "pt": p2CM_decay * np.sqrt(1 - cost**2),
            "pz": p2CM_decay * cost,
            "phi": phiz,
        }
    )
    P3CM_decay = vector.array(
        {
            "E": E3CM_decay,
            "pt": -p2CM_decay * np.sqrt(1 - cost**2),
            "pz": -p2CM_decay * cost,
            "phi": phiz,
        }
    )

    # four-momenta in the LAB frame
    if boost:
        pBoost = np.sqrt(boost["EP_LAB"] ** 2 - m1**2)

        P1LAB_decay = vector.array(
            {
                "E": boost["EP_LAB"],
                "pt": pBoost * np.sqrt(1 - boost["costP_LAB"] ** 2),
                "pz": pBoost * boost["costP_LAB"],
                "phi": boost["phiP_LAB"],
            }
        )

        P2LAB_decay = P2CM_decay.boost_p4(pBoost)
        P3LAB_decay = P3CM_decay.boost_p4(pBoost)

        return P1LAB_decay, P2LAB_decay, P3LAB_decay

    else:
        return P1CM_decay, P2CM_decay, P3CM_decay


# Three body decay
# p1 (k1) --> p2(k2) p3(k3) p4(k4)
def three_body_decay(
    samples, boost=False, m1=1, m2=0, m3=0, m4=0, rng=np.random.random
):

    if not samples:
        raise ValueError("Error! No samples were passed to three_body_decay.")
    else:
        # get sample size of the first item
        sample_size = np.shape(list(samples.values())[0])[0]

    # Mandelstam t = m23^2
    tminus = (m2 + m3) ** 2
    tplus = (m1 - m4) ** 2
    if "unit_t" in samples.keys():
        t = (tplus - tminus) * samples["unit_t"] + tminus
    elif "t" in samples.keys():
        t = samples["t"]
    else:
        t = rng_interval(sample_size, tminus, tplus, rng=rng)

    # Mandelstam u = m_24^2
    # from MATHEMATICA
    uplus = 1 / 4 * (
        ((m1) ** (2) + ((m2) ** (2) + (-1 * (m3) ** (2) + -1 * (m4) ** (2))))
    ) ** (2) * (t) ** (-1) + -1 * (
        (
            (
                (
                    -1 * (m2) ** (2)
                    + 1
                    / 4
                    * (t) ** (-1)
                    * (((m2) ** (2) + (-1 * (m3) ** (2) + t))) ** (2)
                )
            )
            ** (1 / 2)
            + -1
            * (
                (
                    -1 * (m4) ** (2)
                    + 1
                    / 4
                    * (t) ** (-1)
                    * ((-1 * (m1) ** (2) + ((m4) ** (2) + t))) ** (2)
                )
            )
            ** (1 / 2)
        )
    ) ** (
        2
    )
    # from MATHEMATICA
    uminus = 1 / 4 * (
        ((m1) ** (2) + ((m2) ** (2) + (-1 * (m3) ** (2) + -1 * (m4) ** (2))))
    ) ** (2) * (t) ** (-1) + -1 * (
        (
            (
                (
                    -1 * (m2) ** (2)
                    + 1
                    / 4
                    * (t) ** (-1)
                    * (((m2) ** (2) + (-1 * (m3) ** (2) + t))) ** (2)
                )
            )
            ** (1 / 2)
            + (
                (
                    -1 * (m4) ** (2)
                    + 1
                    / 4
                    * (t) ** (-1)
                    * ((-1 * (m1) ** (2) + ((m4) ** (2) + t))) ** (2)
                )
            )
            ** (1 / 2)
        )
    ) ** (
        2
    )
    if "unit_u" in samples.keys():
        u = (uplus - uminus) * samples["unit_u"] + uminus
    elif "u" in samples.keys():
        u = samples["u"]
    else:
        u = rng_interval(sample_size, uminus, uplus, rng=rng)

    # Mandelstam v = m_34^2
    # v = m1**2 + m2**2 + m3**2 + m4**2 - u - t

    # E2CM_decay = (m1**2 + m2**2 - v) / 2.0 / m1
    E3CM_decay = (m1**2 + m3**2 - u) / 2.0 / m1
    E4CM_decay = (m1**2 + m4**2 - t) / 2.0 / m1

    # p2CM_decay = np.sqrt(E2CM_decay * E2CM_decay - m2**2)
    p3CM_decay = np.sqrt(E3CM_decay * E3CM_decay - m3**2)
    p4CM_decay = np.sqrt(E4CM_decay * E4CM_decay - m4**2)

    # Polar angle of P_3
    if "unit_c3" in samples.keys():
        c_theta3 = 2 * samples["unit_c3"] - 1
    elif "c3" in samples.keys():
        c_theta3 = samples["c3"]
    else:
        c_theta3 = rng_interval(sample_size, -1, 1, rng=rng)

    phi3 = rng_interval(sample_size, 0.0, 2 * np.pi, rng=rng)

    # Azimuthal angle of P_4 wrt to P_3 (phi_34)
    if "unit_phi34" in samples.keys():
        phi34 = 2 * np.pi * samples["unit_phi34"]
    elif "phi34" in samples.keys():
        phi34 = samples["phi34"]
    else:
        phi34 = rng_interval(sample_size, 0, 2 * np.pi, rng=rng)

    # polar angle of P_4 wrt to P_3 is a known function of u and v
    c_theta34 = (t + u - m2**2 - m1**2 + 2 * E3CM_decay * E4CM_decay) / (
        2 * p3CM_decay * p4CM_decay
    )

    # p1
    P1CM_decay = vector.array(
        {
            "E": m1 * np.ones(sample_size),
            "px": np.zeros(sample_size),
            "py": np.zeros(sample_size),
            "pz": np.zeros(sample_size),
        }
    )
    # p3
    P3CM_decay = vector.array(
        {
            "E": E3CM_decay * np.ones(sample_size),
            "pt": p3CM_decay * np.sqrt(1 - c_theta3**2) * np.cos(phi3),
            "pz": p3CM_decay * c_theta3,
            "phi": phi3,
        }
    )

    # p4 -- built in the frame where p3 is along z
    P4CM_decay = vector.array(
        {
            "E": E4CM_decay * np.ones(sample_size),
            "pt": p4CM_decay * np.sqrt(1 - c_theta34**2),
            "pz": p4CM_decay * c_theta34,
            "phi": phi34,
        }
    )
    # Now we rotate to the standard rest frame where p3 has the polar angle c_theta3 and azimuthal angle phi3
    P4CM_decay = P4CM_decay.rotateY(np.arccos(c_theta34)).rotateZ(phi3)
    # Cfv.rotationz(
    #     Cfv.rotationy_cos(
    #         c_theta3,
    #         sign=-1,
    #     ),
    #     phi3,
    # )
    # p2

    P2CM_decay = P1CM_decay - P4CM_decay - P3CM_decay

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
        # Transform from CM into the LAB frame
        # Decaying neutrino
        P1LAB_decay = P1CM_decay.boost_p4(p4Boost)
        # Outgoing neutrino
        P2LAB_decay = P2CM_decay.boost_p4(p4Boost)
        # Outgoing lepton minus (3)
        P3LAB_decay = P3CM_decay.boost_p4(p4Boost)
        # Outgoing lepton plus (4)
        P4LAB_decay = P4CM_decay.boost_p4(p4Boost)

        return P1LAB_decay, P2LAB_decay, P3LAB_decay, P4LAB_decay

    # four-momenta in the parent particle rest frame
    else:
        return P1CM_decay, P2CM_decay, P3CM_decay, P4CM_decay
