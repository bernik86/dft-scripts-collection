#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 bernik86.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np


def Fermi(E, mu, T, kb=1.380649e-23):
    return 1 / (1 + np.exp((E - mu) / (kb * T)))


def main(args):
    avtrans_fn = args[1:]

    kB = 1.380649e-23  # J/K
    q = 1.6021e-19  # C
    eV = 1.6021e-19  # J
    T0 = 300  # K

    pre_factor = -T0 * np.pi**2 * kB**2 / (3 * q)

    for fn in avtrans_fn:
        E, tr = np.loadtxt(fn, unpack=True)
        E *= eV  # same as: E = E * eV
        lnT = np.log(tr)

        idx = np.nonzero(E < 0)[-1][-1]

        dTdE = (lnT[idx + 1] - lnT[idx]) / (E[idx + 1] - E[idx])

        print("#######################################")
        print(fn)
        print("Seebeck coefficient:", pre_factor * dTdE)
        print("#######################################")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
