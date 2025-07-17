import numpy as np

from mineut import const


class Material:
    """Pure substances; periodic elements."""

    def __init__(self, density, am, A, Z):
        self.density = density
        self.am = am
        self.Z = Z
        self.A = A
        nq = const.NAvo * self.density / self.am
        self.N = nq * self.A
        self.e = nq * self.Z


class CompositMaterial:
    def __init__(self, table):
        """Compositions of materials.
        fraction is the percentage of it that occupies the total material
        """
        self.density = 0
        self.N = 0
        self.e = 0
        self.Z = 0
        self.A = 0
        for material, fraction in table:
            self.density += material.density * fraction
            self.N += material.N * fraction
            self.e += material.e * fraction
            self.Z += material.Z * fraction
            self.A += material.A * fraction


# class unif(Material):
#     """Alloys/compositions of uniform densities."""

#     def __init__(self, density):
#         super().__init__()
#         self.density = density
#         self.N = self.density / (const.m_avg / const.g_to_GeV)
#         self.e = self.N / 2


# Pre-defined substances

# density in g/cm**3; atomic mass in g/mol
Si = Material(2.329, 28.0855, 28, 14)
WSi2 = Material(9.3, 240.01, 240, 102)
Fe = Material(7.874, 55.845, 56, 26)
Al = Material(2.7, 26.981539, 27, 13)
W = Material(19.3, 183.84, 184, 74)
Cu = Material(8.96, 63.546, 64, 29)
PS = Material(1.05, 104.1, 104, 56)
vacuum = Material(0, 1, 0, 0)

# from CLICdet paper
hcal_CLICdet = CompositMaterial(
    [[Fe, 20 / 26.5], [Al, 0.7 / 26.5], [Cu, 0.1 / 26.5], [PS, 3 / 26.5]]
)
ecal_CLICdet = CompositMaterial([[W, 1.9 / 5.05], [Cu, 2.3 / 5.05], [Si, 0.5 / 5.05]])

OinAir = Material(1.225e-3, 15.9994, 16, 8)  # air density in g/cm**3
NinAir = Material(1.225e-3, 14.0067, 14, 7)
ArinAir = Material(1.225e-3, 39.95, 40, 18)
Air = CompositMaterial(
    [
        [OinAir, 0.2095],
        [NinAir, 0.7812],
        [ArinAir, 0.0093],
    ]
)
