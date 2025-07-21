import matplotlib.patches as patches
import numpy as np

from mineut import plot_tools as pt


def get_gyro_radius(E, B):
    return 3.3e2 * E / B  # cm (E in GeV, B in T)


def get_dtheta(s, R):
    return s / R


def propagate(x0, y0, px0, py0, dtheta, s):
    # r = np.sqrt(x0**2 + y0**2)
    theta_p = np.arctan2(py0, px0)
    p = np.sqrt(px0**2 + py0**2)
    pxf = p * np.cos(theta_p - dtheta)
    pyf = p * np.sin(theta_p - dtheta)

    if dtheta == 0:
        return x0 + s * np.cos(theta_p), y0 + s * np.sin(theta_p), pxf, pyf
    else:
        R = s / dtheta
        # coordinates centered around larmor circle
        x0_prime = R * np.cos(np.pi / 2 + theta_p)
        y0_prime = R * np.sin(np.pi / 2 + theta_p)

        xf_prime = R * np.cos(np.pi / 2 + theta_p - dtheta)
        yf_prime = R * np.sin(np.pi / 2 + theta_p - dtheta)

        dx = xf_prime - x0_prime
        dy = yf_prime - y0_prime

        return x0 + dx, y0 + dy, pxf, pyf


def plot_lattice(df, ax, units=1, draw_center_line=True):
    if draw_center_line:
        ax.plot(df["x"] * units, df["y"] * units, linewidth=0.5, c="black")

    # Minimum size of linear step
    ds = 0.1 * units

    # How tall is the magnet for x-y plane
    magnet_thickness = 1 * units
    n_elements = df.index.size
    ds = 0.1 * units

    for i in list(range(n_elements - 100, n_elements - 8)):
        x, y, s = df["x"][i] * units, df["y"][i] * units, df["L"][i] * units
        px, py = df["px"][i], df["py"][i]
        theta_p = np.arctan2(py, px)
        dtheta = df["ANGLE"][i]
        r_arc = s / dtheta

        if df["L"][i] > 0:
            n_discrete_bend = max(int(s / ds), 30)
            x0, y0, px0, py0 = x, y, px, py
            for j in range(n_discrete_bend):
                xn, yn, pxn, pyn = propagate(
                    x0, y0, px0, py0, dtheta / n_discrete_bend, s / n_discrete_bend
                )
                theta_pn = np.arctan2(pyn, pxn)

                if df["KEYWORD"][i] == "SBEND" or df["KEYWORD"][i] == "RBEND":
                    rect = patches.Rectangle(
                        (x0, y0 - magnet_thickness * np.cos(theta_pn) / 2),
                        width=s / n_discrete_bend,
                        height=magnet_thickness,
                        angle=theta_pn * 180 / np.pi,
                        linewidth=0.5,
                        edgecolor=pt.cblind_safe_wheel[0],
                        facecolor=pt.cblind_safe_wheel[0],
                        zorder=0.5,
                        alpha=1,
                    )
                elif (
                    df["KEYWORD"][i] == "QUADRUPOLE"
                    or df["KEYWORD"][i] == "MULTIPOLE"
                    or df["KEYWORD"][i] == "RCOLLIMATOR"
                ):
                    rect = patches.Rectangle(
                        (x0, y0 - magnet_thickness * np.cos(theta_pn) / 2),
                        width=s / n_discrete_bend,
                        height=magnet_thickness,
                        angle=theta_pn * 180 / np.pi,
                        linewidth=0.5,
                        edgecolor=pt.cblind_safe_wheel[1],
                        facecolor=pt.cblind_safe_wheel[1],
                        zorder=0.51,
                        alpha=1,
                    )
                elif df["KEYWORD"][i] == "DRIFT":
                    rect = patches.Rectangle(
                        (x0, y0 - magnet_thickness * np.cos(theta_pn) / 2),
                        width=s / n_discrete_bend,
                        height=magnet_thickness,
                        angle=theta_pn * 180 / np.pi,
                        linewidth=0.5,
                        edgecolor="lightgrey",
                        facecolor="lightgrey",
                        zorder=0.5,
                        alpha=1,
                    )

                ax.add_patch(rect)
                x0, y0, px0, py0 = xn, yn, pxn, pyn

    return ax
