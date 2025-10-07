import vector
import numpy as np

from mineut import const
from mineut import mudecay_tools as mudec
from mineut import lattice_tools as lt


class MuDecaySimulator:
    def __init__(
        self,
        design,
        lattice=None,
        nuflavor=None,
        cycles=1,
        direction="clockwise",
        n_evals=1e5,
        preloaded_events=None,
        remove_ring_fraction=0,
        NLO=True,
        mudecay_model="NLOmudecay_pol",
    ):
        """
        This class is the main simulation class for muon decays in a lattice.

        Parameters:

            design (dict): Dictionary containing the design parameters.
            nuflavor (str, optional): Neutrino flavor, either 'numu' or 'nuebar'. Defaults to None.
            cycles (float, optional): number of times the muons pass through lattice.
            direction (str, optional): Direction of the beam, either 'left' or 'right'. Defaults to "left".
            n_evals (float, optional): Number of VEGAS evaluations. Defaults to 1e5.
            preloaded_events (dict, optional): Dictionary of preloaded events. Defaults to None.
            remove_ring_fraction (float, optional): Fraction of the ring to be removed.
                    Can be a tuple for beginning and end points to be removed or a float if it's the same.
                    Defaults to 0 (entire ring).
            NLO (bool, optional): Next-to-leading order muon decay (radiative corrections). Defaults to True.
            mudecay_model (str, optional): Model for muon decay. Defaults to "NLOmudecay_pol".

        """

        # Design contains all necessary inputs to specify the muon storage/accelerator
        self.design = design
        self.lattice = lattice

        # Check that design contains all mandatory keys
        mandatory_keys = ["muon_polarization"]
        for key in mandatory_keys:
            if key not in self.design:
                raise KeyError(f"Mandatory key '{key}' missing from design dictionary.")

        self.n_evals = n_evals
        self.preloaded_events = preloaded_events
        self.cycles = cycles
        self.direction = direction

        if isinstance(remove_ring_fraction, float) or isinstance(
            remove_ring_fraction, int
        ):
            self.remove_ring_fraction = [
                (1 - remove_ring_fraction) / 2,
                (1 + remove_ring_fraction) / 2,
            ]
        elif isinstance(remove_ring_fraction, list) or isinstance(
            remove_ring_fraction, tuple
        ):
            self.remove_ring_fraction = [
                (1 - remove_ring_fraction[0]) / 2,
                (1 + remove_ring_fraction[1]) / 2,
            ]

        else:
            print("No remove_ring_fraction specified. Using all ring.")
            self.remove_ring_fraction = [0, 0]

        self.nuflavor = nuflavor
        self.muon_charge = (
            -1 if (self.nuflavor == "numu" or self.nuflavor == "nuebar") else +1
        )
        self.muon_polarization = self.design["muon_polarization"]
        self.NLO = NLO
        self.mudecay_model = mudecay_model

        if abs(self.muon_polarization) > 1:
            raise ValueError(
                f"muon_polarization must be between -1 and 1, found design['muon_polarization'] = {self.muon_polarization}"
            )

        if self.preloaded_events:
            for var in self.preloaded_events.keys():
                setattr(self, var, preloaded_events[var])

    def decay_muons(self):
        """
        Simulates the decay of muons and calculates their positions and velocities.

        If preloaded events are available, the function returns immediately.

        Otherwise, it simulates the decays with self.simulate_decays().

        Attributes:
            preloaded_events (bool): Flag indicating if events are preloaded.
            NLO (bool): Flag indicating if Next-to-Leading Order corrections are used.
            sample_size (int): Number of muon samples.
            pmu (np.ndarray): Array containing the momenta of the muons.
            pnu (np.ndarray): Array containing the momenta of the neutrinos.
            pos (np.ndarray): Array to store the positions of the muons.
            vmu (np.ndarray): Array to store the absolute velocities of the muons.
            momenta (np.ndarray): Array to store the momenta of the neutrinos.
            E (np.ndarray): Array to store the energies of the neutrinos.

        Returns:
            self: The instance of the class with updated attributes.
        """

        if self.preloaded_events:
            return self
        else:

            # Monte Carlo event generation
            Generator = mudec.GeneratorEngine(
                mudecay_model=self.mudecay_model,
                Mparent=const.m_mu,
                Mdaughter=const.m_e,
                NLO=self.NLO,
                muon_polarization=self.muon_polarization,
                muon_charge=self.muon_charge,
                nuflavor=self.nuflavor,
                NINT=20,
                NINT_warmup=10,
                NEVAL=self.n_evals,
                NEVAL_warmup=self.n_evals / 10,
            )

            event_dict = Generator.get_events()  # gets all events for +1 helicity

            # Muon 4-momenta -- at rest
            self.pmu_restframe = event_dict["P_decay_mu"]  # all muon decaying momenta

            # Neutrino 4-momenta
            self.pnu_restframe = event_dict["P_decay_nu"]

            # rest frame variables in integration for ease of reweighting
            self.x_CM = event_dict["x_CM"]
            self.costheta_CM = event_dict["costheta_CM"]

            # Normalized weights. Adds up to 1 for a single muon decay.
            self.weights = event_dict["w_flux"] / np.sum(event_dict["w_flux"])
            self.weights = self.weights[:, np.newaxis]

            # number of events simulated
            self.sample_size = self.pmu_restframe.size

            # Muon decay position (initialized to 0)
            self.pos = vector.array(
                {
                    "x": np.zeros(self.sample_size),
                    "y": np.zeros(self.sample_size),
                    "z": np.zeros(self.sample_size),
                }
            )

    def reweight_with_new_polarization(self, new_polarization):

        numerator = mudec.mudecay_matrix_element_sqr(
            self.x_CM,
            self.costheta_CM,
            new_polarization,
            self.muon_charge,
            self.nuflavor,
            self.NLO,
        )
        denominator = mudec.mudecay_matrix_element_sqr(
            self.x_CM,
            self.costheta_CM,
            self.muon_polarization,
            self.muon_charge,
            self.nuflavor,
            self.NLO,
        )
        # NOTE: Changing polarization does not change the total integral, so this is a safe operation
        self.weights_new_pol = self.weights.flatten() * numerator / denominator

        return self.weights_new_pol

    def place_muons_on_lattice(self, lattice=None, direction="clockwise"):
        """Changes the coordinate axis, position, and momenta of particles to fit a storage ring geometry.

        For the lattice, x is horizontal and y is vertical directions in the transverse plane.

        lattice: dictionary with smooth function that describes the lattice (created from interpolation of .tfs files)

            each function returns the value of the lattice parameter as a function of a parameter `u`:
                `u` goes from 0 to 1, with 0 being the IP and 1 being the IP again after a full central orbit.

            lattice['x']: x position of the muons
            lattice['y']: y position of the muons
            lattice['s']: s displacement of the muons along lattice
            lattice['t']: time of the muons
            lattice['angle_of_central_p']: angle of the central momentum of the muons (tangent to the central orbit)

            lattice['inv_s']: inverse function of s(u) -- it returns u(s) s.t. lattice['inv_s'](lattice['s'](u)) = u

        NOTE: the difference between coordinate systems in MuC and in the lattice

          Lattice reference:
         x -- horizontal (ring) plane
         y -- vertical plane
         z -- longitudinal along motion

         In MuC code:
         x -- normal to the ring (downwards when looking from the IP to the center of the ring)
         y -- horizontal plane (+y exits moves outwards from IP away from the ring)
         z -- along motion tangent to the ring at the IP (+z is a beam from left to right at the IP)

         So in our code, horizontal ~ y, vertical ~ x, and z ~ z

        """

        if isinstance(lattice, dict):
            lattice = lt.Lattice(**lattice)
        elif lattice is None:
            if hasattr(self, "lattice"):
                lattice = self.lattice
            else:
                raise ValueError(
                    "No lattice was set. Please provide a dictionary that describes the lattice or pass a lt.Lattice object."
                )

        # Muon decay position (initialized to origin)
        self.pos = vector.array(
            {
                "x": np.zeros(self.sample_size),
                "y": np.zeros(self.sample_size),
                "z": np.zeros(self.sample_size),
            }
        )

        # get total length of the central orbit
        C = lattice.s(1)  # cm
        
        total_s = self.cycles * C

        # Place muons uniformly along travel path
        self.s_muon = np.random.uniform(0, total_s, self.sample_size)
        # s_in_turn is the position of the muons in the current turn
        self.s_in_turn = (self.s_muon) % C

        # parameter u that goes from 0 to 1 along the lattice
        u_parameter = lattice.inv_s(self.s_in_turn)

        # Get the beam momentum
        beam_p = np.random.normal(
            loc=lattice.beam_p0(u_parameter),
            scale=lattice.beamdiv_z(u_parameter) * lattice.beam_p0(u_parameter),
            size=self.sample_size,
        )

        # Set the muon 4-momenta along lattice
        self.pmu = vector.array(
            {
                "E": np.sqrt(beam_p**2 + const.m_mu**2),
                "px": np.zeros(self.sample_size),
                "py": np.zeros(self.sample_size),
                "pz": beam_p,
            }
        )

        # Adding the beam transverse divergence
        # Handle beam divergence (can be function or constant)

        # Rotation in 2D commutes, so as long as we only rotation in transverse plane
        theta_x = np.random.normal(
            loc=0.0,
            scale=lattice.beamdiv_x(u_parameter),
            size=self.sample_size,
        )
        theta_y = np.random.normal(
            loc=0.0,
            scale=lattice.beamdiv_y(u_parameter),
            size=self.sample_size,
        )

        # Rotate by beam divergence envelope
        self.pmu = self.pmu.rotateX(-np.arctan(theta_x))
        self.pmu = self.pmu.rotateY(np.arctan(theta_y))

        # Boost the neutrino 4 momenta
        self.pnu = self.pnu_restframe.boost_p4(self.pmu)

        # Absolute velocity of muons
        self.vmu = const.c_LIGHT * np.sqrt((1 - (const.m_mu / self.pmu["E"]) ** 2))
        
        # spread muons in time according to number of beam lifetimes desired
        self.mutimes = self.s_muon / self.vmu  # time in seconds
        max_time = total_s / self.vmu #final time

        self.muon_lifetime = const.tau0_mu * self.pmu["E"] / const.m_mu
        print("pmu_E: ",self.pmu["E"])

        # Now, if we want to increase our efficiency in the simulation, we better force particles to be close to the detector in some way.
        # Let's enforce this by clipping the s_in_turn range:
        zacc_min = C * self.remove_ring_fraction[0]
        zacc_max = C * self.remove_ring_fraction[1]
        events_likely_within_acceptance = (self.s_in_turn <= zacc_min) | (
            self.s_in_turn > zacc_max
        )

        # Now we move muons along an extra distance to fall within acceptance, paying the price of a smaller survival probability
        shifted_events = ~events_likely_within_acceptance
        deltaz_min = zacc_max - self.s_in_turn
        deltaz_max = C - self.s_in_turn + zacc_min
        shift_z = np.zeros(self.sample_size)
        shift_z[shifted_events] = (
            np.random.uniform(0, 1, shifted_events.sum())
            * (deltaz_max[shifted_events] - deltaz_min[shifted_events])
            + deltaz_min[shifted_events]
        )

        # Apply shift to s_in_this_turn and to the travel time of the muons
        self.s_in_turn = self.s_in_turn + shift_z
        self.mutimes += shift_z / self.vmu
        self.s_in_turn = self.s_in_turn % C

        # Acceptance of simulated region
        #self.weights[:, 0] = self.weights[:, 0] * (zacc_min + (C - zacc_max)) / C

        print("before:", sum(self.weights[:, 0]))
        # Apply exponential suppression on total length travelled by muons
        self.weights[:, 0] *= (
            # 1 - np.exp(-self.mutimes / self.muon_lifetime)
            lattice.Nmu_per_bunch*(max_time)*np.exp(-self.mutimes / self.muon_lifetime) / self.muon_lifetime
        )

        print("max time: ", max_time)
        print("number of samples: ", self.sample_size)
        print("total s: ", total_s)
        print("delta t: ", max_time/self.sample_size)
        print("Nmu per bunch: ", lattice.Nmu_per_bunch)
        print("muon lifetime: ", self.muon_lifetime)
        print("mu times: ",self.mutimes)
        print("after:", sum(self.weights[:, 0]))

        # Now deform locations to real space along the lattice

        # put everyone in the z axis
        self.pos["z"] = self.s_in_turn

        # Straight outta lattice parameterization
        self.pos["z"] = lattice.x(u_parameter)
        self.pos["y"] = lattice.y(u_parameter)
        self.pos["x"] = np.zeros(self.sample_size)

        # Rotate to central orbit
        theta_central_orbit = lattice.angle_of_central_p(u_parameter)
        self.pnu = self.pnu.rotateX(-theta_central_orbit)
        self.pmu = self.pmu.rotateX(-theta_central_orbit)

        # Rotation in 2D commutes, so as long as we only rotation in transverse plane
        x_horizontal = np.random.normal(
            loc=0.0,
            scale=lattice.beamsize_x(u_parameter),
            size=self.sample_size,
        )
        x_vertical = np.random.normal(
            loc=0.0,
            scale=lattice.beamsize_y(u_parameter),
            size=self.sample_size,
        )

        # Now shift coordinates by location of beam envelope
        # vertical component is trivial
        self.pos["x"] = self.pos["x"] + x_vertical
        self.pos["z"] = self.pos["z"] + x_horizontal * np.sin(theta_central_orbit)
        self.pos["y"] = self.pos["y"] + x_horizontal * np.cos(theta_central_orbit)

        # Shift time so t = 0 is bunch crossing (NOTE: mutimes will always be negative.)
        t_per_turn = C / self.vmu
        time_in_this_turn = self.mutimes % t_per_turn
        self.mutimes_to_bunchx = np.where(
            time_in_this_turn < t_per_turn / 2,  # in first half of turn
            time_in_this_turn,  # time wrt bunch xs is + (past crossing)
            time_in_this_turn - t_per_turn,  # time wrt bunch xs is - (future crossing)
        )

        # if right moving, then mirror trajectories through y axis
        if direction == "counter-clockwise":
            self.pos["x"] = -1 * self.pos["x"]
            self.pnu = self.pnu.rotateX(np.pi)
            self.pmu = self.pmu.rotateX(np.pi)

        return self

    def get_flux_at_generic_location(
        self,
        det_location=[0, 0, 1e5],
        det_radius=1e2,
        ebins=100,
        acceptance=False,
        per_area=True,
        new_polarization=None,
        normalization=1,
        mask=None,
    ):

        if mask is None:
            mask = np.ones(self.sample_size, dtype=bool)
        det_location = np.asarray(det_location)
        if det_location.shape != (3,):
            raise ValueError("det_location must be a list or array of length 3.")
        det_vector = vector.array(
            {"x": det_location[0], "y": det_location[1], "z": det_location[2]}
        )
        # normal_to_detector_plane = det_vector.unit()
        distances = det_vector - self.pos[mask]
        neutrino_direction = self.pnu[mask].to_3D().unit()
        dotprod = distances.dot(neutrino_direction)

        # Project distance vector onto neutrino direction
        sintheta = np.sqrt(1 - distances.unit().dot(neutrino_direction) ** 2)

        # Position of closest approach on the neutrino path
        radial_distance = sintheta * distances.mag

        # Check if neutrino crosses the detector disk
        in_acceptance = (dotprod > 0) & (radial_distance < det_radius)

        # Detector area
        area = np.pi * det_radius**2

        # Select appropriate weights
        if new_polarization is not None:
            self.reweight_with_new_polarization(new_polarization)
            w = self.weights_new_pol[mask]
        else:
            w = self.weights[mask, 0]

        # Normalize accepted neutrino weights
        if np.sum(w) == 0:
            print("Warning: sum of weights is zero.")
            return 0, 0
        nu_eff_ND = np.sum(w[in_acceptance]) / np.sum(w)

        if acceptance:
            print("Detector acceptance: {:.2e}".format(nu_eff_ND))
            return nu_eff_ND

        if nu_eff_ND > 0:
            Enu_ND, flux_nu_ND_p = get_flux(
                self.pnu["E"][mask][in_acceptance], w[in_acceptance], ebins
            )

            # Convert to flux per unit area per unit energy
            if per_area:
                flux_nu_ND = normalization * flux_nu_ND_p / area / np.diff(Enu_ND)
            else:
                flux_nu_ND = normalization * flux_nu_ND_p / np.diff(Enu_ND)

            return Enu_ND, flux_nu_ND
        else:
            print("No flux through detector.")
            return 0, 0

    def get_acceptance_map_fixed_z(
        self,
        z_location=293062,         # about 100m away
        xrange=(-5000,5000), #match yrange to make square
        yrange=(-77245, 10e2),      #about 10m above and below
        nx=50,                       # grid resolution (x)
        ny=50,                       # grid resolution (y)
        det_radius=1e2               # detector radius
    ):
        """
        Generate a 2D map of the neutrino count (acceptance) at a fixed z-plane.

        Parameters:
            z_location  - Fixed z where detectors are placed (cm)
            xrange, yrange - Spatial ranges for X and Y
            nx, ny      - Number of grid points along x and y
            det_radius  - Detector radius in cm

        Returns:
            X, Y, acceptance_map
        """
        # Create grid points
        x_vals = np.linspace(xrange[0], xrange[1], nx)
        y_vals = np.linspace(yrange[0], yrange[1], ny)

        # Initialize map for detector acceptance
        acceptance_map = np.zeros((ny, nx))  # y is row, x is column

        # Loop over detector positions
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                nu_eff_ND = self.get_flux_at_generic_location(
                    det_location=[x, y, z_location],
                    det_radius=det_radius,
                    acceptance=True,  #Just count neutrinos, no energy binning
                )

                acceptance_map[i, j] = nu_eff_ND if nu_eff_ND is not None else 0.0

        # Create meshgrid for plotting
        X, Y = np.meshgrid(x_vals, y_vals)
        return X, Y, acceptance_map


# class BINSimulator:
#     """
#       Main class of MuC library

#       Simulates all beam-induced neutrino interactions in a given MuC detector.

#       Hierarchy is as follows:
#           Initializes by creating instances of DecaySimulation.

#           self.run(), runs simulations of many SimNeutrinos based on the collision type,
#     which is a single-neutrino-species MC generation of events within a detector, which are all saved in a list in the .sims attribute.
#     """

#     def __init__(
#         self,
#         design,
#         n_evals=1e5,
#         preloaded_events=None,
#         det_geom="det_v2",
#         save_mem=True,
#         lattice=None,
#         remove_ring_fraction=0,
#     ):
#         """Initializes a BIN simulation for a given detector geometry and collider lattice"""

#         self.save_mem = save_mem
#         self.design = design
#         self.det_geom = getattr(mineut, det_geom)
#         if isinstance(remove_ring_fraction, float) or isinstance(
#             remove_ring_fraction, int
#         ):
#             self.remove_ring_fraction = [
#                 (1 - remove_ring_fraction) / 2,
#                 (1 + remove_ring_fraction) / 2,
#             ]
#         elif isinstance(remove_ring_fraction, list) or isinstance(
#             remove_ring_fraction, tuple
#         ):
#             self.remove_ring_fraction = [
#                 (1 - remove_ring_fraction[0]) / 2,
#                 (1 + remove_ring_fraction[1]) / 2,
#             ]

#         else:
#             print("No remove_ring_fraction specified. Using all ring.")
#             self.remove_ring_fraction = [0, 0]

#         self.lattice = lattice
#         if isinstance(self.lattice, str):
#             try:
#                 with open(lattice, "rb") as f:
#                     self.lattice = pickle.load(f)
#             except Exception as errormessage:
#                 print(errormessage)
#                 raise ValueError(f"Could not load lattice from file: {lattice}.")
#         elif isinstance(self.lattice, dict):
#             self.lattice = lattice
#             assert (
#                 "x" in self.lattice.keys()
#             ), f"Need x(u) in lattice dictionary. Found keys: {self.lattice.keys()}"
#             assert (
#                 "inv_s" in self.lattice.keys()
#             ), "Need inverse function of s(u) in lattice dictionary."
#         else:
#             print("No lattice specified. Using simplified ring geometry.")
#             self.lattice = None

#         # Total length of the ring in cm
#         self.C = self.lattice["s"](1) if self.lattice is not None else self.design["C"]
#         self.beam_lifetime = 1 / self.design["finj"]
#         self.n_turns = self.beam_lifetime / (self.C / const.c_LIGHT)
#         self.bunchx_in_a_year = (
#             self.n_turns
#             * self.design["duty_factor"]
#             * self.design["finj"]
#             * self.design["bunch_multiplicity"]
#             * np.pi
#             * 1e7  # seconds in a year
#         )

#         # Detector geometry
#         self.comps = list(self.det_geom.facedict.keys())
#         self.zending = self.det_geom.zending
#         self.rmax = self.det_geom.rmax

#         self.n_evals = n_evals

#         # Container for all simulations
#         self.mustorage_sims = []
#         self.beam_cases = col.colls_types_to_beam_cases[self.design["collision_type"]]
#         self.nuflavors = [part[0] for part in self.beam_cases]
#         for nuflavor, direction in self.beam_cases:
#             self.mustorage_sims.append(
#                 MuDecaySimulator(
#                     design=design,
#                     nuflavor=nuflavor,
#                     direction=direction,
#                     n_evals=n_evals,
#                     beam_lifetime=self.beam_lifetime,
#                     preloaded_events=preloaded_events,
#                     remove_ring_fraction=self.remove_ring_fraction,
#                 )
#             )

#         # Number of simulations
#         self.nsims = len(self.mustorage_sims)

#     # @profile
#     def run(
#         self,
#         show_components=0,
#         show_time=0,
#     ):
#         """Runs the whole simulation, based on a storage ring geometry, detector geometry, and collision.

#         Args:
#             show_components (bool): for distribution within different detector components.
#             show_time (bool): for time summary.
#             geom (str): the detector version. Latest is det_v2; uniform block is block_test; zero_density_test is exactly that; and det_v1 is simpler.
#             Lss (float): Length of the straight segment upon which the IR detector is centered.
#         """

#         # Attempts to perform a muon decay simulation
#         self.sims = []
#         for mu_sim in self.mustorage_sims:

#             # Decay muons
#             mu_sim.decay_muons()

#             # Place muon along the collider ring
#             if self.lattice is not None:
#                 mu_sim.place_muons_on_lattice(
#                     direction="left",  # mu_sim.direction,
#                     lattice=self.lattice,
#                 )
#             else:
#                 mu_sim.place_muons_on_simplified_ring(
#                     C=self.C,
#                     Lss=self.design["Lss"],
#                     direction="left",  # mu_sim.direction,
#                 )
#             # Now onto the detector simulations
#             det_sim = DetectorSimulator(mu_sim, self.det_geom, save_mem=self.save_mem)
#             det_sim.run()

#             # NOTE: This weight is per bunch
#             det_sim.w *= (
#                 self.design["finj"]
#                 * self.design["duty_factor"]
#                 * self.design["bunch_multiplicity"]
#                 * np.pi
#                 * 1e7  # s in a year
#             )

#             det_sim.calculate_facecounts()

#             if self.save_mem:
#                 det_sim.clear_mem()

#             self.sims.append(det_sim)

#         self.total_count = np.sum([np.sum(self.sims[i].w) for i in range(self.nsims)])

#         self.name = self.design["name"]
#         # print(f"Successfully simulated neutrino event rates within {self.geom.name}:")
#         # print(
#         # f"{self.name} ({col.acc_colls_dict[self.design['collision_type']]}) at L = {self.design['Lss']:.2f} m."
#         # )
#         print(f"Total count: {self.total_count:.2e} events;\n")

#         self.facecounts = self.get_face_counts()
#         self.get_exclusive_rates()

#         if show_components:
#             self.print_face_counts()

#         if show_time:
#             self.print_timetable()

#         return self

#     def reweight_with_new_polarization(self, new_polarization):

#         for sim in self.sims:
#             sim.calculate_facecounts_new_pol(new_polarization)
#         return self.get_face_counts_new_pol()

#     def get_exclusive_rates(self):
#         """
#         Calculate and aggregate exclusive rates for all neutrino species.
#         This method performs the following steps:
#         1. Initializes an empty dictionary `self.exclusive_rates`.
#         2. Iterates over all simulations in `self.sims` and retrieves their exclusive rates.
#         3. Aggregates the exclusive rates for each neutrino flavor, component, and channel.
#         4. Initializes an empty dictionary `self.exclusive_rates_combined`.
#         5. Aggregates the exclusive rates for ECAL and HCAL components across all neutrino species.
#         6. Ensures that all test flavors ("nue", "numu", "nuebar", "numubar") have entries in `self.exclusive_rates_combined`, initializing them to 0 if they do not exist.
#         The resulting exclusive rates are stored in `self.exclusive_rates` and `self.exclusive_rates_combined`.
#         Returns:
#             None
#         """

#         # Append exclusive rates all neutrino species
#         self.exclusive_rates = {}
#         for s in self.sims:
#             s.get_exclusive_rates()
#             for (comp, channel), rate in s.get_exclusive_rates().items():
#                 key = s.nuflavor, comp, channel.replace(s.nuflavor + "_", "")
#                 if key in self.exclusive_rates.keys():
#                     self.exclusive_rates[key] += rate
#                 else:
#                     self.exclusive_rates[key] = rate

#         # Exclusive rates from all neutrino species within ECAL and HCAL
#         self.exclusive_rates_combined = {}
#         for (flavor, comp, channel), rate in self.exclusive_rates.items():
#             if comp == "hcal" or comp == "ecal":
#                 if (flavor, channel) in self.exclusive_rates_combined.keys():
#                     self.exclusive_rates_combined[flavor, channel] += rate
#                 else:
#                     self.exclusive_rates_combined[flavor, channel] = rate
#         for test_flavor in ["nue", "numu", "nuebar", "numubar"]:
#             for (flavor, comp, channel), rate in self.exclusive_rates.items():
#                 if (test_flavor, channel) in self.exclusive_rates_combined.keys():
#                     continue
#                 else:
#                     self.exclusive_rates_combined[test_flavor, channel] = 0

#     def event_timing(
#         self,
#         fs=(20, 12),
#         histtype="barstacked",
#         nbins=100,
#         savefig=None,
#         legend=False,
#         title=True,
#         sec="all",
#         nuflavor="all",
#     ):
#         """Wrapper to plot neutrino interaction times.

#         Args:
#             savefig (str): name of file to save plot to.
#             fs (tuple): figsize of plot. Can be None to display on the same plot that is being worked on.
#             title (bool): if one wants to display the pre-generated title.
#             legend (bool): to display the pre-generated legend.
#             sec (str): the component of the detector one wants to single out. Options are the sames as those written in get_data() method description.
#             nuflavor (str): a particle one would want to single out. Can be either nue, nuebar, numu, or numubar.
#         """

#         if fs:
#             plt.figure(figsize=fs)

#         _, _, _, w, times, _, _ = self.get_data(sec=sec, nuflavor=nuflavor)

#         label = f"{sec}; " + r"$N_{events}$" + f": {np.sum(w):.3e}"

#         if sec == "all":
#             label = r"$L_{ss} = $" + f"{self.design['Lss']:.0f} m"

#         times *= 1e9

#         plt.xlabel("Time before collision (ns)")
#         plt.ylabel(r"$N_{events}$")

#         if title:
#             plt.title(
#                 f"Event Timing (wrt bunch crossing); ({self.name} at L = {self.design['Lss']:.2f})"
#             )

#         plt.hist(times, weights=w, histtype=histtype, bins=nbins, label=label)

#         if legend:
#             plt.legend(loc="best")

#         plt.yscale("log")

#         if savefig:
#             plt.savefig(savefig, bbox_inches="tight", dpi=300)

#     def get_GENIE_flux(self, sec, nuflavor, nbins=50):
#         """
#         Calculate the GENIE flux histogram for a given section and neutrino flavor.

#         Parameters:
#         sec (int): Detector component to consider.
#         nuflavor (str): Neutrino flavor. Can be either nue, nuebar, numu, or numubar.
#         nbins (int, optional): The number of bins for the histogram. Default is 50.

#         Returns:
#         tuple: A tuple containing the bin edges and the histogram values.
#         """
#         _, _, _, w, _, E, _ = self.get_data(sec=sec, nuflavor=nuflavor, genie=1)
#         h = np.histogram(E, weights=w, bins=nbins)
#         return h[1], h[0]

#     def print_GENIE_flux_to_file(self, sec, nuflavor, nbins=50, filename=None):
#         """
#             Creates a flux .data file for GENIE simulation of events.

#             sec (str): Detector component to consider.
#             nuflavor (str): Neutrino flavor. Can be either nue, nuebar, numu, or numubar.
#             nbins (int, optional): Number of bins for the histogram. Default is 50.
#             filename (str, optional): Name of file to be saved in the fluxes/ folder. If not provided, a default name will be generated.

#         Returns:
#             None

#         Saves:
#             A .data file containing the flux information for GENIE simulation.
#         """

#         if filename:
#             fn = f"{filename}"

#         else:
#             fn = f"fluxes/{self.design['short_name']}_{mineut.compsto2[sec]}_{nuflavor}.data"

#         bins, flux = self.get_GENIE_flux(sec, nuflavor, nbins=nbins)
#         bin_centers = bins[:-1] + np.diff(bins) / 2
#         np.savetxt(fn, np.array([bin_centers, flux]).T)
#         print(f"Flux file saved to {fn}")

#     def plot_GENIE_flux(self, sec, nuflavor, nbins=50, ax=None):
#         """
#         Plots the GENIE flux for a given neutrino flavor and sector.

#         Parameters:
#             sec (str): The detector section for which the flux is to be plotted.
#             nuflavor (str): Neutrino flavor. Can be either nue, nuebar, numu, or numubar.
#             nbins (int), optional: The number of bins to use for the histogram (default is 50).
#             ax (matplotlib.axes.Axes), optional: The axes on which to plot the histogram. If None, a new figure and axes are created (default is None).

#         Returns:
#             None
#         """

#         bins, flux = self.get_GENIE_flux(sec=sec, nuflavor=nuflavor, nbins=nbins)
#         if ax is None:
#             fig, ax = plt.subplots()
#         _ = ax.hist(
#             bins[:-1] + np.diff(bins) / 2,
#             weights=flux,
#             bins=bins,
#             histtype="step",
#             label=nuflavor + "_" + sec,
#         )

#     def get_GENIE_event_weight(self, comp, p, n_events):
#         """Getting the weight of a genie particle from its detector component."""
#         try:
#             return (
#                 self.facecounts[comp, p + "_left"] / n_events
#             )  # the last factor depends on how many generated events there are in the files. It only supports same n files across detectors.
#         except KeyError:
#             return self.facecounts[comp, p + "_right"] / n_events  #
#         except KeyError:
#             return 0

#     def load_GENIE_file(self, filename, n_events=1e5):
#         """Loads a single GENIE analysis file, adding respective weights based on this simulation's number of BIN interactions.

#         Args:
#             filename (str): name of the GENIE file to load.
#             n_events: number of events that the GENIE file had.

#         """

#         with open(filename, "r") as file:

#             for i, line in enumerate(file):

#                 if i == 0:
#                     continue

#                 elif i == 1:
#                     exp = (line.split(":")[1])[:-1]

#                 elif i == 2:
#                     particles = ((line.split(":")[1])[:-1]).split(",")

#                 elif i == 3:
#                     comps = (line.split(":")[1])[:-1]

#                 else:
#                     break

#         expname = exp
#         parts = [mineut.partn_names[part] for part in particles]

#         particlenames = ", ".join(parts)
#         t = comps.replace(",", ", ")

#         print(f"Loading generated data for a {expname} experiment;")
#         print(
#             f"It includes interactions from {particlenames} within the {t} of the muon detector."
#         )

#         data = pd.read_csv(filename, sep=r"\s+", skiprows=5)

#         print("Adding weights...")
#         try:
#             data["w"] = data.apply(
#                 lambda row: self.get_GENIE_event_weight(
#                     row["DComp"], mineut.pdg2names[str(row["IncL"])], n_events=n_events
#                 ),
#                 axis=1,
#             )
#         except KeyError:
#             data["w"] = data.apply(
#                 lambda row: self.get_GENIE_event_weight(
#                     row["DComp"], str(row["Particle"]), n_events=n_events
#                 ),
#                 axis=1,
#             )

#         if "Q2" in data.columns:
#             data["Q2"] = get_Q2(data["nu_E"], data["E"], data["pz"])

#         print("Done!")

#         return data

#     def load_genie_events(self, filenames, n_events=1e6):
#         """
#         Load GENIE events from the specified filenames.
#         Parameters:
#         filenames (str or list of str): The filename(s) from which to load GENIE events.
#                                         Can be a single filename or a list of filenames.
#         n_events (int, optional): The number of events to load. Default is 1e6.

#         Returns: None
#             This method sets the following attributes:
#                 - genie_events: A DataFrame containing the loaded GENIE events.
#                 - genie_e: A boolean array indicating electron events.
#                 - genie_mu: A boolean array indicating muon events.
#                 - genie_tau: A boolean array indicating tau events.
#                 - genie_nue: A boolean array indicating electron neutrino events.
#                 - genie_numu: A boolean array indicating muon neutrino events.
#                 - genie_nutau: A boolean array indicating tau neutrino events.
#                 - genie_nuebar: A boolean array indicating electron antineutrino events.
#                 - genie_numubar: A boolean array indicating muon antineutrino events.
#                 - genie_nutaubar: A boolean array indicating tau antineutrino events.`
#         """

#         if isinstance(filenames, list):
#             data_cases = []
#             for filename in filenames:
#                 data_cases.append(
#                     self.load_GENIE_file(f"{filename}", n_events=n_events)
#                 )
#             self.genie_events = pd.concat(data_cases, axis=0)
#         else:
#             self.genie_events = self.load_GENIE_file(f"{filenames}", n_events=n_events)

#         try:
#             self.genie_e = np.abs(self.genie_events["OutL"]) == 11
#             self.genie_mu = np.abs(self.genie_events["OutL"]) == 13
#             self.genie_tau = np.abs(self.genie_events["OutL"]) == 15

#             self.genie_nue = self.genie_events["IncL"] == 12
#             self.genie_numu = self.genie_events["IncL"] == 14
#             self.genie_nutau = self.genie_events["IncL"] == 16

#             self.genie_nuebar = self.genie_events["IncL"] == -12
#             self.genie_numubar = self.genie_events["IncL"] == -14
#             self.genie_nutaubar = self.genie_events["IncL"] == -16
#         except KeyError:
#             self.genie_e = (self.genie_events["Name"] == "e-") | (
#                 self.genie_events["Name"] == "e+"
#             )
#             self.genie_mu = (self.genie_events["Name"] == "mu-") | (
#                 self.genie_events["Name"] == "mu+"
#             )
#             self.genie_tau = (self.genie_events["Name"] == "tau-") | (
#                 self.genie_events["Name"] == "tau+"
#             )

#             self.genie_nue = self.genie_events["Particle"] == "nue"
#             self.genie_numu = self.genie_events["Particle"] == "numu"
#             self.genie_nutau = self.genie_events["Particle"] == "nutau"

#             self.genie_nuebar = self.genie_events["Particle"] == "nuebar"
#             self.genie_numubar = self.genie_events["Particle"] == "numubar"
#             self.genie_nutaubar = self.genie_events["Particle"] == "nutaubar"

#         return self.genie_events


def get_Q2(nu_E, E, pz):
    """Getting Q squared from the generated events."""
    return -1 * ((nu_E - E) ** 2 - (nu_E - pz) ** 2)


def get_flux(x, w, nbins):
    hist1 = np.histogram(
        x, weights=w, bins=nbins, density=False, range=(np.min(x), np.max(x))
    )

    ans0 = hist1[1]
    ans1 = hist1[0]  # /(ans0[1]-ans0[0])
    return ans0, ans1
