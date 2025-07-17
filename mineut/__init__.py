import importlib

from mineut import const
from mineut import mudecay_tools
from mineut import detector_tools
from mineut import collider_tools
from mineut import lattice_tools
from mineut import MuC

# Relevant dictionaries for treating sim demands.
anti_neutrinos = ["nuebar", "numubar"]
neutrinos = ["numu", "nue"]
part_names = {"nue": "ν_e", "nuebar": "anti ν_e", "numu": "ν_μ", "numubar": "anti ν_μ"}
partn_names = {"12": "ν_e", "-12": "anti ν_e", "14": "ν_μ", "-14": "anti ν_μ"}


directions = ["left", "left", "right", "right"]
compsto2 = {
    "muon_detector": "MD",
    "solenoid_borders": "SB",
    "solenoid_mid": "SM",
    "hcal": "HC",
    "ecal": "EC",
    "nozzles": "NO",
}
comps_short_to_long = {
    "Total": "Total",
    "MD": "muon_detector",
    "SB": "solenoid_borders",
    "SM": "solenoid_mid",
    "HC": "hcal",
    "EC": "ecal",
    "NO": "nozzles",
}
pdg2names = {
    "12": "nue",
    "-12": "nuebar",
    "14": "numu",
    "-14": "numubar",
    "16": "nutau",
    "-16": "nutaubar",
}
