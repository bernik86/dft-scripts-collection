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
import argparse

from pyscf import tools, lib


def main(args):
    parser = argparse.ArgumentParser(
        description="Generate cube files from Pyscf MO coefficients for specified orbitals"
    )
    parser.add_argument("-chk", required=True, help="Pyscf checkfile")
    parser.add_argument(
        "-iorbs",
        nargs="+",
        help="Indices of orbitals for which WFs are generated (First orbital has index 1)",
    )
    parser.add_argument("-res", default=0.5, type=float, help="Resolution (Bohr)")
    parser.add_argument(
        "-pf",
        default="",
        help="Add specified postfix to cube filename (before file ending)",
    )
    args = parser.parse_args()

    chk = args.chk
    orbs = args.iorbs
    res = args.res
    pf = args.pf

    mol = lib.chkfile.load_mol(chk)
    print(mol.atom)
    print(mol.charge)
    print(mol.spin)
    print(mol.basis)

    mo_coeff = lib.chkfile.load("checkfile", "scf/mo_coeff")

    orbs = map(int, orbs)
    print(orbs)

    for i in orbs:
        print("generating orbital ", i)
        if mo_coeff.shape[0] == 2 and len(mo_coeff.shape) == 3:
            tools.cubegen.orbital(
                mol,
                "orbital_{:04d}_{}_alpha.cube".format(i, pf),
                mo_coeff[0, :, i - 1],
                resolution=res,
            )
            tools.cubegen.orbital(
                mol,
                "orbital_{:04d}_{}_beta.cube".format(i, pf),
                mo_coeff[1, :, i - 1],
                resolution=res,
            )
        else:
            tools.cubegen.orbital(
                mol,
                "orbital_{:04d}_{}.cube".format(i, pf),
                mo_coeff[:, i - 1],
                resolution=res,
            )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
