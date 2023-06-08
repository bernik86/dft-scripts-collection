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
import argparse

import numpy as np


def idx_gen(lines, phrase, offset=0):
    return (i + offset for i, s in enumerate(lines) if s.find(phrase) > -1)


def get_natoms(lines):
    idx = idx_gen(lines, "Molecular Point Group", -2)
    try:
        index = next(idx)
    except StopIteration:
        n_atoms = 0
    else:
        n_atoms = int(lines[index].split()[0].strip())

    return n_atoms


def print_array(array, digits=4):
    fmt = "%10." + digits + "f"
    np.savetxt(sys.stdout.buffer, array, fmt=fmt, delimiter=" ")
    sys.stdout.flush()


def get_nacs(lines, index, n_atoms, etf=False, gs=False, force=False):
    offset = 12
    if not gs:
        if etf:
            offset = offset + 2 * n_atoms + 10
    else:
        if not etf:
            offset = offset + 2 * n_atoms + 10

    if force:
        offset = 12 + n_atoms + 5

    nacs = "".join(lines[index + offset : index + offset + n_atoms])
    nacs = np.fromstring(nacs, sep=" ")
    nacs = nacs.reshape(n_atoms, -1)[:, 1:]

    return nacs


def print_nacs(
    nacs_etf,
    nacs,
    norms_only=False,
    response=False,
    digits=4,
    print_minimal=False,
    force=False,
):
    norms_only = norms_only or print_minimal
    fmt = "{:." + digits + "f}"
    if response:
        if not print_minimal:
            print(
                "Quadratic response: ||τ|| = " + fmt.format(np.linalg.norm(nacs_etf)),
                flush=True,
            )
    elif force:
        print("||h|| (a.u.) = " + fmt.format(np.linalg.norm(nacs_etf)), flush=True)
    else:
        if not print_minimal:
            print(
                "With ETF: ||τ|| = " + fmt.format(np.linalg.norm(nacs_etf)), flush=True
            )
    if not norms_only:
        print_array(nacs_etf, digits)

    if nacs is not None:
        if not print_minimal:
            print(
                "Without ETF: ||τ|| = " + fmt.format(np.linalg.norm(nacs)), flush=True
            )
        if not norms_only:
            print_array(nacs, digits)

    if print_minimal:
        print(
            fmt.format(np.linalg.norm(nacs_etf)), "\t", fmt.format(np.linalg.norm(nacs))
        )


def main(args):
    parser = argparse.ArgumentParser(description="Extract NACs")
    parser.add_argument("--all", action="store_true", help="Extract all NACs")
    parser.add_argument("--norms-only", action="store_true")
    parser.add_argument("--nac-force", action="store_true")
    parser.add_argument(
        "--response", action="store_true", help="Print nacs from response theory"
    )
    parser.add_argument(
        "--print-minimal",
        action="store_true",
        help="Only print numbers: NAC_etf NAC_no_etf",
    )
    parser.add_argument("-i", default="0", help="Initial state")
    parser.add_argument("-f", default="1", help="Final state")
    parser.add_argument("-o", required=True, help="Output file")
    parser.add_argument(
        "-n",
        type=int,
        default=2,
        help="Number of atoms, in case automatic determination fails",
    )
    parser.add_argument("-d", default="4", help="Number of digits printed")
    args = parser.parse_args()

    open_file = open
    np.set_printoptions(suppress=True, precision=4, floatmode="fixed")

    if args.o.endswith(".bz2"):
        open_file = bz2.open

    with open_file(args.o, "rt") as output_file:
        lines = output_file.readlines()

    idx = idx_gen(lines, "              between states", 0)

    n_atoms = get_natoms(lines)
    print(n_atoms)
    if n_atoms == 0:
        n_atoms = args.n

    found_nac = False

    while not found_nac:
        try:
            index = next(idx)
        except StopIteration:
            found_nac = True
            continue

        states = lines[index].split()[2::2]
        initial, final = states[0] == args.i, states[1] == args.f

        if initial and final or args.all:
            ground_state = int(states[0]) == 0
            if not args.print_minimal:
                print("Coupling: {} -> {}".format(states[0], states[1]))
            if args.response:
                nacs_etf = get_nacs(lines, index, n_atoms, False, gs=ground_state)
                nacs = None
            elif args.nac_force:
                nacs_etf = get_nacs(
                    lines, index, n_atoms, False, gs=ground_state, force=True
                )
                nacs = None
            else:
                nacs_etf = get_nacs(lines, index, n_atoms, True, gs=ground_state)
                nacs = get_nacs(lines, index, n_atoms, False, gs=ground_state)
            print_nacs(
                nacs_etf,
                nacs,
                args.norms_only,
                args.response,
                args.d,
                args.print_minimal,
                force=args.nac_force,
            )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
