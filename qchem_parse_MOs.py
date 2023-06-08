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
import bz2

import numpy as np

from qchem_extract_nacs import idx_gen


def load_MOs(outp, C, start_idx, n_basis):
    offset = n_basis + 2

    for i in range(int(n_basis / 6)):
        C[:, i * 6 : i * 6 + 6] = np.loadtxt(
            outp,
            usecols=(4, 5, 6, 7, 8, 9),
            skiprows=start_idx + i * offset,
            max_rows=n_basis,
        )


def main(args):
    with bz2.open(args[1], "rt", encoding="utf-8") as outp_file:
        lines = outp_file.readlines()

    idx = next(idx_gen(lines, "basis functions"))
    n_basis = int(lines[idx].strip().split()[-3])
    print(n_basis)

    C = {}
    C["alpha"] = np.zeros((n_basis, n_basis))
    C["beta"] = np.zeros((n_basis, n_basis))

    idx_MO = idx_gen(lines, "MOLECULAR ORBITAL COEFFICIENTS")

    load_MOs(args[1], C["alpha"], next(idx_MO) + 3, n_basis)
    load_MOs(args[1], C["beta"], next(idx_MO) + 3, n_basis)

    for spin in C:
        np.savetxt("MOs_" + spin, C[spin], fmt="%.7f")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
