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
from dataclasses import dataclass, field
from typing import List

import numpy as np
from ase.units import Hartree

from qchem_extract_nacs import idx_gen


def init_transmom():
    return np.zeros((3))


def fluorescence(state):
    c = 137  # find more accurate
    t_au = 2.4188843265857e-17
    E = state.excitation_energy / Hartree
    mu = np.linalg.norm(state.trans_moment)

    k = 4 * E**3 * mu**2 / (3 * c**3)
    t = t_au / k

    print("{:e} {:e}".format(k, t))


@dataclass
class ExcitedState:
    number: int = -1
    energy: float = -1.0
    excitation_energy: float = -1.0
    strength: float = -1.0
    trans_moment: np.array = field(default_factory=init_transmom)
    transitions: dict = field(default_factory=dict)
    main_transition: str = ""

    def _parse_tddft(self, lines: List[str]):
        self.number = int(lines[0].split()[2].strip(":"))
        self.excitation_energy = float(lines[0].split("=")[-1].strip())
        self.energy = float(lines[1].split()[-2].strip(":"))
        if lines[2].strip().find("Multiplicity") > -1:
            mult_offset = 0
        else:
            mult_offset = -1
        trans_mom = lines[3 + mult_offset].strip().split()
        i = 0
        for tm in trans_mom:
            try:
                tm_i = float(tm)
            except ValueError:
                continue
            else:
                self.trans_moment[i] = tm_i
                i += 1
        self.strength = float(lines[4 + mult_offset].strip().split(" ")[-1])
        for line in lines[5 + mult_offset :]:
            # trans = line.strip().split(':')[1].strip().split('amplitude =')
            trans = line.strip().split("amplitude =")
            self.transitions[trans[0].strip()] = float(trans[1].split()[0].strip())
            if self.main_transition == "":
                self.main_transition = trans[0].strip()

    def _parse_eomcc(self, lines):
        self.number = int(lines[0].split()[2].split("/")[0])
        self.energy = float(lines[1].split()[3])
        for line in lines[::-1]:
            if line.find("Transitions between orbitals") > -1:
                break
            trans = line.split(maxsplit=1)
            self.transitions[trans[1].strip()] = float(trans[0].strip())

    def parse_state(self, lines: List[str], is_cc: bool = False) -> None:
        if is_cc:
            self._parse_eomcc(lines)
        else:
            self._parse_tddft(lines)

    def set_main_transition(self, trans):
        if trans in self.transitions:
            self.main_transition = trans
        else:
            raise KeyError("State does not have transition: " + trans)

    def __gt__(self, other):
        return np.fabs(self.transitions[self.main_transition]) > np.fabs(
            other.transitions[other.main_transition]
        )

    def __eq__(self, other):
        return (
            self.transitions[self.main_transition]
            == other.transitions[other.main_transition]
        )

    def __repr__(self):
        rep = [
            "State number: " + str(self.number),
            "State energy (a.u.): " + str(self.energy),
            "Excitation energy (eV): " + str(self.excitation_energy),
            "Transition moment: " + str(self.trans_moment),
            "Oscillator strength: " + str(self.strength),
        ]

        for key in self.transitions:
            rep.append(key + ":\t{:8.4f}".format(self.transitions[key]))

        return "\n".join(rep)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", required=True, help="Filename output file")
    parser.add_argument(
        "-t", default="D(    2) --> V(    1)", help="Expression of transistion"
    )
    parser.add_argument("-f", help="Load expression of transistion from file")
    parser.add_argument(
        "--only-state-number",
        action="store_true",
        help="Only show excited state number",
    )
    parser.add_argument(
        "--print-minimal", action="store_true", help="print state number and energy"
    )
    parser.add_argument("--print-all", action="store_true")
    parser.add_argument("--print-norm", action="store_true")
    parser.add_argument("--cc", action="store_true", help="EOM-CC")
    parser.add_argument("-s", default=0, type=int, help="Line offset (useful for GOs)")
    parser.add_argument(
        "--fluorescence",
        action="store_true",
        help="Calculate fluorescence rate and lifetime",
    )
    args = parser.parse_args()

    open_file = open
    if args.o.endswith(".bz2"):
        open_file = bz2.open

    with open_file(args.o, "rt") as output_file:
        lines = output_file.readlines()

    if args.f is not None:
        with open(args.f, "rt") as transition_file:
            args.t = transition_file.readline().strip()

    start_tddft_gen = idx_gen(lines, "TDDFT Excitation Energies")
    if args.cc:
        start_tddft = idx_gen(lines, "Davidson procedure converged")

    while (start_tddft := next(start_tddft_gen)) < args.s:
        pass

    print(start_tddft)
    idx_state = (
        i + start_tddft
        for i, s in enumerate(lines[start_tddft:])
        if s.find(args.t) > -1
    )
    idx_state_start = (
        i + start_tddft
        for i, s in enumerate(lines[start_tddft:])
        if s.find("excitation energy (eV)") > -1
    )

    idx_state_start = np.array(list(idx_state_start))

    states = []
    for idx in idx_state:
        start_idx = np.argwhere(idx_state_start < idx).reshape(-1)[-1]
        start_line = idx_state_start[start_idx]
        if start_idx + 1 < len(idx_state_start):
            end_line = idx_state_start[start_idx + 1]
        else:
            temp_idx = start_line
            while not lines[temp_idx].strip():
                temp_idx += 1
            end_line = temp_idx - 1
        state = []
        excited_state = ExcitedState()
        state = lines[start_line : end_line - 1]
        if len(state) > 0:
            excited_state.parse_state(state, args.cc)
            excited_state.set_main_transition(args.t)
            states.append(excited_state)

    final_state = max(states, default=ExcitedState())

    if args.only_state_number:
        print(final_state.number)
    elif args.print_minimal:
        state_nr = final_state.number
        state_energy = final_state.energy
        osc = final_state.strength
        print(state_energy, "\t", state_nr, "\t", osc)
    elif args.print_all:
        print("#####################################")
        print("Number of states found:", len(states))
        for st in states:
            print(st)
    else:
        print(final_state)
        if args.print_norm:
            X = []
            Y = []
            for trans, ampl in final_state.transitions.items():
                if trans.startswith("X:"):
                    X.append(ampl)
                elif trans.startswith("Y:"):
                    Y.append(ampl)
            if X:
                print("||X||: ", np.linalg.norm(np.array(X)))
                print("Sum X", np.sum(np.array(X)))
            if Y:
                print("||Y||: ", np.linalg.norm(np.array(Y)))
                print("Sum Y", np.sum(np.array(Y)))

        if args.fluorescence:
            fluorescence(final_state)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
