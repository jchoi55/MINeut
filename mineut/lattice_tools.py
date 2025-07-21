import pandas as pd
import numpy as np
import matplotlib.patches as patches
from scipy.interpolate import interp1d

from mineut import const
from mineut import plot_tools as pt


class Lattice:
    """
    Class to represent a particle accelerator lattice.
    """

    def __init__(self, **kwargs):

        lattice_dict = kwargs.copy()

        for key in [
            "beam_p0",
            "x",
            "y",
            "s",
            "angle_of_central_p",
            "inv_s",
        ]:
            if key in lattice_dict:
                setattr(self, key, lattice_dict.pop(key))
            else:
                raise ValueError(f"Lattice dictionary must contain {key} key.")

        # Transverse beam divergence
        if "beamdiv_x" in lattice_dict and "beamdiv_y" in lattice_dict:
            self.beamdiv_x = lattice_dict.pop("beamdiv_x")
            self.beamdiv_y = lattice_dict.pop("beamdiv_y")
        elif "beamdiv" in lattice_dict:
            self.beamdiv_x = lattice_dict.pop("beamdiv")
            self.beamdiv_y = self.beamdiv_x
        elif (
            ("beamdiv_x" in lattice_dict and "beamdiv_y" not in lattice_dict)
            or ("beamdiv_x" not in lattice_dict and "beamdiv_y" in lattice_dict)
            or ("beamdiv_x" in lattice_dict and "beam_div" in lattice_dict)
            or ("beamdiv_y" in lattice_dict and "beam_div" in lattice_dict)
        ):
            raise ValueError("Inconsistent beam divergence specifications.")
        else:
            self.beamdiv_x = lambda x: 0
            self.beamdiv_y = lambda x: 0

        # Longitudinal beam divergence
        if "beamdiv_z" in lattice_dict:
            self.beamdiv_z = lattice_dict.pop("beamdiv_z")
        else:
            self.beamdiv_z = lambda x: 0

        # Adding the beam transverse size
        if "beamsize_x" in lattice_dict and "beamsize_y" in lattice_dict:
            self.beamsize_x = lattice_dict.pop("beamsize_x")
            self.beamsize_y = lattice_dict.pop("beamsize_y")
        elif "beamsize" in lattice_dict:
            self.beamsize_x = lattice_dict.pop("beamsize")
            self.beamsize_y = self.beamsize_x
        elif (
            ("beamsize_x" in lattice_dict and "beamsize_y" not in lattice_dict)
            or ("beamsize_x" not in lattice_dict and "beamsize_y" in lattice_dict)
            or ("beamsize_x" in lattice_dict and "beamsize" in lattice_dict)
            or ("beamsize_y" in lattice_dict and "beamsize" in lattice_dict)
        ):
            raise ValueError("Inconsistent beam size specifications.")
        else:
            self.beamsize_x = lambda x: 0
            self.beamsize_y = lambda x: 0

        # Longitudinal beam size
        if "beamsize_z" in lattice_dict:
            self.beamsize_z = lattice_dict.pop("beamsize_z")
        else:
            self.beamsize_z = lambda x: 0

        # Making sure everyone is callable
        for attr in [
            "beam_p0",
            "beamdiv_x",
            "beamdiv_y",
            "beamdiv_z",
            "beamsize_x",
            "beamsize_y",
            "beamsize_z",
        ]:
            val = getattr(self, attr)
            if not callable(val):
                if isinstance(val, (int, float)):
                    setattr(self, attr, lambda _, v=val: v)
                else:
                    print(
                        "Warning! Attribute",
                        attr,
                        "is not callable and not a number. Setting it to zero.",
                    )
                    setattr(self, attr, lambda _, v=val: 0)

        self.Nmu_per_bunch = lattice_dict.pop("Nmu_per_bunch", 1)
        self.duty_factor = lattice_dict.pop("duty_factor", 1)
        self.bunch_multiplicity = lattice_dict.pop("bunch_multiplicity", 1)
        self.finj = lattice_dict.pop("finj", 1)

        for key, value in lattice_dict.items():
            print("Setting additional", key, "to", value)
            self.__setattr__(key, value)
        if lattice_dict.keys():
            print(
                "Warning! The following keys were not recognized and will be ignored:",
                lattice_dict.keys(),
            )


def create_racetrack_lattice(
    straight_length=100e2, total_length=300e2, n_elements=10_000, **kwargs
):

    racetrack_radius = (
        (total_length - 2 * straight_length) / 2 / np.pi
    )  # radius of the semicircles
    n_points = 300
    # Semicircle on the left
    theta_left = np.linspace(3 * np.pi / 2, np.pi / 2, int(n_points / 4))
    x_left = -straight_length / 2 + racetrack_radius * np.cos(theta_left)
    y_left = racetrack_radius * np.sin(theta_left)

    # Semicircle on the right
    theta_right = np.linspace(np.pi / 2, 3 * np.pi / 2, int(n_points / 4))
    x_right = straight_length / 2 - racetrack_radius * np.cos(theta_right)
    y_right = racetrack_radius * np.sin(theta_right)

    # Top straight
    x_top = np.linspace(-straight_length / 2, straight_length / 2, int(n_points / 4))
    y_top = np.full_like(x_top, racetrack_radius)

    # Bottom straight
    x_bottom = np.linspace(straight_length / 2, -straight_length / 2, int(n_points / 4))
    y_bottom = np.full_like(x_bottom, -racetrack_radius)

    # Concatenate all segments
    x_racetrack = np.concatenate([x_left, x_top, x_right, x_bottom])
    y_racetrack = np.concatenate([y_left, y_top, y_right, y_bottom])
    y_racetrack -= np.max(y_racetrack)

    lattice_dict = create_lattice_dict_from_vertices(
        (x_racetrack, y_racetrack), n_elements=n_elements
    )
    # Any additional user-input
    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    lattice.vertices = (x_racetrack, y_racetrack)

    return lattice

def create_straight_lattice(
    total_length=100e2, n_elements=10_000, **kwargs
):

    n_points = 300

    #Straight
    x_track= np.linspace(-total_length / 2, total_length / 2, int(n_points / 4))
    y_track = np.full_like(x_track, 0)

    lattice_dict = create_lattice_dict_from_vertices(
        (x_track, y_track), n_elements=n_elements
    )
    # Any additional user-input
    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    lattice.vertices = (x_track, y_track)

    return lattice

def create_dogbone_lattice(
    straight_length=100e2, total_length=300e2, m = 2, n_elements=10_000, **kwargs
):
    #for now this is found using mathematica
    scale = 1946.5338408838027

    n_points = 300
    theta= np.linspace(0, 2 * np.pi, int(n_points / 4))
    # Teardrop on the left
    x_left = -straight_length / 2 + scale*(np.cos(theta)-1)
    y_left = scale*np.sin(theta)*(np.sin(theta*0.5))**m

    # Teardrop on the right
    x_right = straight_length / 2 - scale*(np.cos(theta)-1)
    y_right = scale*np.sin(theta)*(np.sin(theta*0.5))**m

    # Straight
    x_track = np.linspace(straight_length / 2, -straight_length / 2, int(n_points / 4))
    y_track = np.full_like(x_track, 0)

    # Concatenate all segments
    x_racetrack = np.concatenate([x_left, x_track, x_right])
    y_racetrack = np.concatenate([y_left, y_track, y_right])
    #y_racetrack -= np.max(y_racetrack)

    lattice_dict = create_lattice_dict_from_vertices(
        (x_racetrack, y_racetrack), n_elements=n_elements
    )
    # Any additional user-input
    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    lattice.vertices = (x_racetrack, y_racetrack)

    return lattice

def create_elliptical_lattice(
    length_minor, length_major, center=(0, 0), n_elements=10_000, **kwargs
):
    """Create an elliptical lattice.

    Args:
        length_minor (_type_): _description_
        length_major (_type_): _description_
        n_elements (_type_, optional): _description_. Defaults to 10_000.
    """
    theta = np.linspace(0, 2 * np.pi, n_elements)
    x_ellipse = center[0] + length_major * np.cos(theta)
    y_ellipse = center[1] + length_minor * np.sin(theta)

    lattice_dict = create_lattice_dict_from_vertices(
        (x_ellipse, y_ellipse), n_elements=n_elements
    )

    # Any additional user-input
    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    lattice.vertices = (x_ellipse, y_ellipse)

    return lattice


def get_gyro_radius(E, B):
    return 3.3e2 * E / B  # cm (E in GeV, B in T)


def get_dtheta(s, R):
    return s / R


def advance_in_pos_and_momentum(x0, y0, px0, py0, dtheta, ds):
    # r = np.sqrt(x0**2 + y0**2)
    theta_p = np.arctan2(py0, px0)
    p = np.sqrt(px0**2 + py0**2)
    pxf = p * np.cos(theta_p - dtheta)
    pyf = p * np.sin(theta_p - dtheta)

    if dtheta == 0:
        return x0 + ds * np.cos(theta_p), y0 + ds * np.sin(theta_p), pxf, pyf
    else:
        R = ds / dtheta
        # coordinates centered around larmor circle
        x0_prime = R * np.cos(np.pi / 2 + theta_p)
        y0_prime = R * np.sin(np.pi / 2 + theta_p)

        xf_prime = R * np.cos(np.pi / 2 + theta_p - dtheta)
        yf_prime = R * np.sin(np.pi / 2 + theta_p - dtheta)

        dx = xf_prime - x0_prime
        dy = yf_prime - y0_prime

        return x0 + dx, y0 + dy, pxf, pyf


def advance_in_pos(x0, y0, theta_0, dtheta, ds):
    theta_mid = theta_0 + dtheta / 2
    x_new = x0 + ds * np.cos(theta_mid)
    y_new = y0 + ds * np.sin(theta_mid)
    return x_new, y_new


def create_lattice_dict_from_vertices(vertices, n_elements=None):
    x_points, y_points = vertices
    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)

    if n_elements is None:
        n_elements = len(x_points)
    else:
        n_elements = min(max(n_elements, len(x_points)), int(1e6))

    # Compute arc-length (s) along the curve
    dx = np.diff(x_points)
    dy = np.diff(y_points)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    s_vals = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = s_vals[-1]

    # Create interpolation functions
    fx = interp1d(s_vals, x_points, kind="linear")
    fy = interp1d(s_vals, y_points, kind="linear")

    # New arc-length positions for smooth sampling
    s_dense = np.linspace(0, total_length, n_elements)

    x_dense = fx(s_dense)
    y_dense = fy(s_dense)
    angle_dense = np.arctan2(np.gradient(y_dense), np.gradient(x_dense))

    # Normalize to u ∈ [0, 1]
    u = np.linspace(0, 1, n_elements)

    lattice_dict = {
        "x": interp1d(u, x_dense, bounds_error=True),
        "y": interp1d(u, y_dense, bounds_error=True),
        "s": interp1d(u, s_dense, bounds_error=True),
        "angle_of_central_p": interp1d(u, angle_dense, bounds_error=True),
        "inv_s": interp1d(s_dense, u, bounds_error=True),
    }

    return lattice_dict


def get_lattice_dataframe_from_tfs(filename):
    # Initialize lists to store metadata and column data
    metadata = {}
    columns = []
    data = []

    # Open and read file
    with open(filename, "r") as file:
        for line in file:
            # Extract metadata lines
            if line.startswith("@"):
                parts = line.split()
                key = parts[1]
                value = " ".join(parts[3:]).strip('"')
                metadata[key] = value
            # Extract column names
            elif line.startswith("*"):
                columns = line.strip().split()[
                    1:
                ]  # Strip and split, and ignore the '*' character
            # Skip format line
            elif line.startswith("$"):
                continue
            # Extract data lines
            else:
                fields = [s.strip('"') for s in line.strip().split()]

                data.append(fields)

    # Validate and filter correct data rows
    correct_data = [row for row in data if len(row) == len(columns)]
    df = pd.DataFrame(correct_data, columns=columns)

    # Convert numerical columns to appropriate data types
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass

    n_elements = df.index.size

    new_cols = {
        "bending_magnet": df["ANGLE"] != 0,
        "GAMMAX": (1 + df["ALFX"] ** 2) / df["BETX"],
        "GAMMAY": (1 + df["ALFY"] ** 2) / df["BETY"],
        "x": np.zeros((n_elements,)),
        "y": np.zeros((n_elements,)),
        "px": np.zeros((n_elements,)),
        "py": np.zeros((n_elements,)),
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Assign metadata to the DataFrame's attributes
    df.attrs = metadata

    # Put the initial conditions for the particle momentum
    df.loc[0, "px"] = float(df.attrs["ENERGY"])

    for i in range(0, n_elements):
        if df["L"][i] == 0:
            (
                df.loc[i + 1, "x"],
                df.loc[i + 1, "y"],
                df.loc[i + 1, "px"],
                df.loc[i + 1, "py"],
            ) = (df["x"][i], df["y"][i], df["px"][i], df["py"][i])
            continue
        else:
            (
                df.loc[i + 1, "x"],
                df.loc[i + 1, "y"],
                df.loc[i + 1, "px"],
                df.loc[i + 1, "py"],
            ) = advance_in_pos_and_momentum(
                df["x"][i],
                df["y"][i],
                df["px"][i],
                df["py"][i],
                dtheta=df["ANGLE"][i],
                ds=df["L"][i],
            )

    return df


# ds in meters
def create_smoothed_lattice(df, emittance_RMS=1e-6, n_elements=None, **kwargs):
    """
    Create a smooth lattice representation from an existing lattice DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the lattice data with columns 'x', 'y', 'L', 'px', 'py', 'ANGLE', 'BETX', 'BETY', 'GAMMAX', 'GAMMAY', 'DX', 'DPX'.
    emittance_RMS : float
        RMS emittance value to calculate beam sizes.
    n_elements : int
        Number of elements in the new smoother lattice.
    Returns
    -------
    lattice_dict : dict
        Dictionary containing smoothed lattice data with keys 'x', 'y', 's', 'angle_of_central_p', 'beamsize_x', 'beamsize_y',
        'beamdiv_x', 'beamdiv_y', 'dispersion_Dx', 'dispersion_Dpx', 'inv_s'.
    """

    n_elements_current = df.index.size

    if n_elements <= n_elements_current:
        print(
            f"Warning: n_elements ({n_elements}) is less than or equal to the current number of elements ({n_elements_current})."
        )
        n_elements = n_elements_current

    # element of length in new smoother lattice
    ds = df["S"].max() / n_elements

    # All desired units are cm or seconds or radians
    smooth_curve_x = np.array([])
    smooth_curve_y = np.array([])

    smooth_curve_s = np.array([])

    smooth_curve_angle_of_central_p = np.array([])

    smooth_curve_beamsize_x = np.array([])
    smooth_curve_beamsize_y = np.array([])

    smooth_curve_beamdiv_x = np.array([])
    smooth_curve_beamdiv_y = np.array([])

    smooth_curve_dispersion_Dx = np.array([])
    smooth_curve_dispersion_Dpx = np.array([])

    for i in range(0, n_elements_current):
        # for i in list(range(n_elements_current-200,n_elements_current)):
        x, y, ell = df["x"][i], df["y"][i], df["L"][i]
        s = df["S"][i] - ell
        px, py = df["px"][i], df["py"][i]
        dtheta = df["ANGLE"][i]
        # theta_p = np.arctan2(py, px)
        # r_arc = l / dtheta

        if df["L"][i] > 0:
            n_discrete_bend = int(ell / ds)
            if n_discrete_bend < 1:
                n_discrete_bend = 1
            x0, y0, px0, py0 = x, y, px, py
            for j in range(n_discrete_bend):
                xn, yn, pxn, pyn = advance_in_pos_and_momentum(
                    x0, y0, px0, py0, dtheta / n_discrete_bend, ell / n_discrete_bend
                )
                theta_pn = np.arctan2(pyn, pxn)

                smooth_curve_s = np.append(smooth_curve_s, (s + ds * j) * const.m_to_cm)
                smooth_curve_x = np.append(smooth_curve_x, xn * const.m_to_cm)
                smooth_curve_y = np.append(smooth_curve_y, yn * const.m_to_cm)
                smooth_curve_angle_of_central_p = np.append(
                    smooth_curve_angle_of_central_p, theta_pn
                )

                smooth_curve_beamsize_x = np.append(
                    smooth_curve_beamsize_x,
                    np.sqrt(emittance_RMS * df["BETX"][i]) * const.m_to_cm,
                )
                smooth_curve_beamsize_y = np.append(
                    smooth_curve_beamsize_y,
                    np.sqrt(emittance_RMS * df["BETY"][i]) * const.m_to_cm,
                )

                smooth_curve_beamdiv_x = np.append(
                    smooth_curve_beamdiv_x,
                    np.arctan(np.sqrt(emittance_RMS * df["GAMMAX"][i])),
                )
                smooth_curve_beamdiv_y = np.append(
                    smooth_curve_beamdiv_y,
                    np.arctan(np.sqrt(emittance_RMS * df["GAMMAY"][i])),
                )

                smooth_curve_dispersion_Dx = np.append(
                    smooth_curve_dispersion_Dx, df["DX"][i]
                )
                smooth_curve_dispersion_Dpx = np.append(
                    smooth_curve_dispersion_Dpx, df["DPX"][i]
                )

                x0, y0, px0, py0 = xn, yn, pxn, pyn

    lattice_dict = {}
    u = np.linspace(0, 1, len(smooth_curve_s))
    lattice_dict["x"] = interp1d(u, smooth_curve_x, bounds_error=False, fill_value=None)
    lattice_dict["y"] = interp1d(u, smooth_curve_y, bounds_error=False, fill_value=None)
    lattice_dict["s"] = interp1d(u, smooth_curve_s, bounds_error=False, fill_value=None)

    lattice_dict["angle_of_central_p"] = interp1d(
        u, smooth_curve_angle_of_central_p, bounds_error=False, fill_value=None
    )
    lattice_dict["beamsize_x"] = interp1d(
        u, smooth_curve_beamsize_x, bounds_error=False, fill_value=None
    )
    lattice_dict["beamsize_y"] = interp1d(
        u, smooth_curve_beamsize_y, bounds_error=False, fill_value=None
    )
    lattice_dict["beamdiv_x"] = interp1d(
        u, smooth_curve_beamdiv_x, bounds_error=False, fill_value=None
    )
    lattice_dict["beamdiv_y"] = interp1d(
        u, smooth_curve_beamdiv_y, bounds_error=False, fill_value=None
    )
    lattice_dict["dispersion_Dx"] = interp1d(
        u, smooth_curve_dispersion_Dx, bounds_error=False, fill_value=None
    )
    lattice_dict["dispersion_Dpx"] = interp1d(
        u, smooth_curve_dispersion_Dpx, bounds_error=False, fill_value=None
    )

    lattice_dict["inv_s"] = interp1d(
        smooth_curve_s, u, bounds_error=False, fill_value=None
    )

    lattice_dict["beam_p0"] = np.sqrt(
        float(df.attrs["ENERGY"]) ** 2 - const.m_mu**2
    )  # Default to 1.0 if not set

    # Update with user input
    lattice_dict.update(kwargs)

    return lattice_dict


def plot_lattice(df):
    fig, ax = pt.std_fig(figsize=(10, 5))
    # ax.set_xlim(-220, 0)
    # ax.set_xlim(125, 150)
    # ax.set_ylim(-11, 11)
    # ax.scatter(df['x'][~df['bending_magnet']], df['y'][~df['bending_magnet']], marker='|', s=200, color='darkorange', zorder=2)
    # ax.scatter(df['x'][df['bending_magnet']], df['y'][df['bending_magnet']], marker='x', s=200, color='dodgerblue', zorder=2)
    ax.plot(df["x"], df["y"], linewidth=0.5, c="black")

    rect = patches.Rectangle(
        (-6, -6),
        12,
        12,
        linewidth=2,
        edgecolor="black",
        facecolor="None",
        hatch="///////",
    )
    ax.add_patch(rect)

    # Minimum size of linear step
    ds = 0.1
    # How tall is the magnet for x-y plane
    magnet_thickness = 1
    n_elements = df.index.size
    ds = 0.1
    # for i in list(range(1,100))+list(range(n_elements-100,n_elements)):
    for i in list(range(n_elements - 400, n_elements)):
        x, y, s = df["x"][i], df["y"][i], df["L"][i]
        px, py = df["px"][i], df["py"][i]
        dtheta = df["ANGLE"][i]
        # theta_p = np.arctan2(py, px)
        # r_arc = s / dtheta

        if df["L"][i] > 0:
            n_discrete_bend = max(int(s / ds), 30)
            x0, y0, px0, py0 = x, y, px, py
            for j in range(n_discrete_bend):
                xn, yn, pxn, pyn = advance_in_pos_and_momentum(
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
                        edgecolor="dodgerblue",
                        facecolor="dodgerblue",
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
                        edgecolor="orange",
                        facecolor="orange",
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

    ax.set_ylim(df["y"].min(), 10)
    ax.set_xlim(df["x"].min(), 0)

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    # if df['KEYWORD'][i] == 'DRIFT':
    # ax.plot([x, x+s*np.cos(theta_p)], [y, y+s*np.sin(theta_p)], color='black', linewidth=2)

    fig.savefig(
        f'plots/beam_optics/lattice_{df.attrs["ENERGY"]}_trajectory.pdf',
        dpi=500,
        bbox_inches="tight",
    )
