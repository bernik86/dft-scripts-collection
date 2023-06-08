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
import bz2

import numpy as np

from qchem_extract_nacs import idx_gen


def read_energies(lines, vdw=False, return_np=False, TDA=False):
    idx = idx_gen(lines, "SCF   energy in the final basis set")
    if vdw:
        for gs_idx in idx:
            pass
    else:
        gs_idx = next(idx)
    E_gs_au = lines[gs_idx].strip().split()[-1]

    es_idx = [i for i, s in enumerate(lines) if s.find("Total energy for state") > -1]
    tddft_start = next(idx_gen(lines, "TDDFT Excitation Energies"))
    if TDA:
        es_tddft_ix = [i for i in es_idx if i < tddft_start]
    else:
        es_tddft_ix = [i for i in es_idx if i > tddft_start]

    E_es_au = [lines[i].strip().split()[-2] for i in es_tddft_ix]
    E_au = [E_gs_au] + E_es_au

    if return_np:
        E_au = np.array(E_au, dtype=float)

    return E_au


def read_osc(lines, return_np=False, TDA=False):
    es_idx = [i for i, s in enumerate(lines) if s.find("Strength   :") > -1]
    tddft_start = next(idx_gen(lines, "TDDFT Excitation Energies"))
    if TDA:
        es_tddft_ix = [i for i in es_idx if i < tddft_start]
    else:
        es_tddft_ix = [i for i in es_idx if i > tddft_start]
    E_au = [lines[i].strip().split()[-1] for i in es_tddft_ix]

    if return_np:
        E_au = np.array(E_au, dtype=float)

    return E_au


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", required=True, help="QChem output file")
    parser.add_argument("-n", type=int, default=0, help="# of states (0=all)")
    parser.add_argument(
        "-S", type=int, default=-1, help="only state with certain number"
    )
    parser.add_argument("--align", action="store_true", help="Set GS energy to 0 a.u.")
    parser.add_argument(
        "--osc", action="store_true", help="Print oscillator strength instead of energy"
    )
    parser.add_argument(
        "--vdw", action="store_true", help="Whether calculation was performed with vdW"
    )
    args = parser.parse_args()

    with bz2.open(args.o, "rt") as f_outp:
        lines = f_outp.readlines()

    if args.osc:
        E_au = read_osc(lines)
    else:
        E_au = read_energies(lines, args.vdw)

    if args.n:
        E_au = E_au[: args.n]
    elif args.S > -1:
        E_au = [E_au[args.S]]

    if args.align:
        E_au = [str(float(i) - float(E_au[0])) for i in E_au]

    print("\t".join(E_au))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
