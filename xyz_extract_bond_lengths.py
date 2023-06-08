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

import numpy as np
from ase import atoms

enc = "utf-8"


def bonds(b_fn, coords, types, form_str, autosave):
    natoms = coords.shape[0]
    if b_fn is not None:
        bonds = [
            (int(l.split(",")[0]), int(l.split(",")[1]))
            for i, l in enumerate(open(b_fn, "rt", encoding=enc).readlines())
        ]
    else:
        bonds = [(i, i + 1) for i in range(1, natoms)]
        if autosave:
            with open("autosave_bonds.txt", "wt", encoding=enc) as f:
                for i, (a1, a2) in enumerate(bonds):
                    f.write("{}, {}\n".format(a1, a2))
    print("# Bonds: ", bonds)

    for i, (a1, a2) in enumerate(bonds):
        bv = coords[a1 - 1] - coords[a2 - 1]
        bl = np.linalg.norm(bv)
        print(form_str.format(types[a1 - 1], a1, types[a2 - 1], a2, bl))


def avg_bonds(b_fn, coords, types, form_str, autosave):
    if b_fn is not None:
        bonds = [
            (str(l.split(",")[0]), tuple(l.strip("\n").split(",")[1:]))
            for i, l in enumerate(open(b_fn, "rt", encoding=enc).readlines())
        ]
    print("# Averaged Bonds: ", bonds)

    for i, (label, ats) in enumerate(bonds):
        n_ats = len(ats)
        if n_ats % 2 != 0:
            print("ERROR: Need even number of atoms!")
            sys.exit(-1)
        norm = n_ats / 2
        avg = 0
        for j in range(0, n_ats, 2):
            a1 = int(ats[j].strip())
            a2 = int(ats[j + 1].strip())
            bv = coords[a1 - 1] - coords[a2 - 1]
            bl = np.linalg.norm(bv)
            avg += bl
        avg /= norm
        print(form_str.format(label, avg))


def angles(a_fn, coords, types, form_str_angle, autosave):
    natoms = coords.shape[0]
    if a_fn is not None:
        angles = [
            (int(l.split(",")[0]), int(l.split(",")[1]), int(l.split(",")[2]))
            for i, l in enumerate(open(a_fn, "rt", encoding=enc).readlines())
        ]
    else:
        angles = [(i, i + 1, i + 2) for i in range(1, natoms - 1)]
        if autosave:
            with open("autosave_angles.txt", "wt", encoding=enc) as f:
                for i, (a1, a2, a3) in enumerate(angles):
                    f.write("{}, {}, {}\n".format(a1, a2, a3))
    print("# Angles", angles)

    for i, (a1, a2, a3) in enumerate(angles):
        v1 = coords[a1 - 1] - coords[a2 - 1]
        v1_l = np.linalg.norm(v1)
        v2 = coords[a3 - 1] - coords[a2 - 1]
        v2_l = np.linalg.norm(v2)
        ang = np.arccos(np.dot(v1, v2) / (v1_l * v2_l))
        ang_deg = ang * 180 / np.pi
        print(
            form_str_angle.format(
                types[a1 - 1], a1, types[a2 - 1], a2, types[a3 - 1], a3, ang_deg
            )
        )


def avg_angles(a_fn, coords, types, form_str_angle, autosave):
    if a_fn is not None:
        angles = [
            (str(l.split(",")[0]), tuple(l.strip("\n").split(",")[1:]))
            for i, l in enumerate(open(a_fn, "rt", encoding=enc).readlines())
        ]
    print("# Averaged Angles", angles)

    for i, (label, angs) in enumerate(angles):
        n_ang = len(angs)
        if n_ang % 3 != 0:
            print("ERROR: Need number of atoms divisible by 3!")
            sys.exit(-1)

        avg = 0
        norm = n_ang / 3
        for j in range(0, n_ang, 3):
            a1 = int(angs[j].strip())
            a2 = int(angs[j + 1].strip())
            a3 = int(angs[j + 2].strip())

            v1 = coords[a1 - 1] - coords[a2 - 1]
            v1_l = np.linalg.norm(v1)
            v2 = coords[a3 - 1] - coords[a2 - 1]
            v2_l = np.linalg.norm(v2)
            ang = np.arccos(np.dot(v1, v2) / (v1_l * v2_l))
            ang_deg = ang * 180 / np.pi
            avg += ang_deg
        avg /= norm
        print(form_str_angle.format(label, avg))


def dihedrals(
    t_fn, coords, types, form_str_angle, autosave, return_values=False, corr_da=False
):
    natoms = coords.shape[0]
    if natoms < 4:
        print("Not enough atoms to calculate dihedral angles!")
        return

    if t_fn is not None:
        angles = [
            (
                int(l.split(",")[0]),
                int(l.split(",")[1]),
                int(l.split(",")[2]),
                int(l.split(",")[3]),
            )
            for i, l in enumerate(open(t_fn, "rt", encoding=enc).readlines())
        ]
    else:
        angles = [(i, i + 1, i + 2, i + 3) for i in range(1, natoms - 2)]
        if autosave:
            with open("autosave_dihedrals.txt", "wt", encoding=enc) as f:
                for i, (a1, a2, a3, a4) in enumerate(angles):
                    f.write("{}, {}, {}, {}\n".format(a1, a2, a3, a4))
    if not return_values:
        print("# Dihedrals", angles)

    res = []
    for i, (a1, a2, a3, a4) in enumerate(angles):
        v1a = coords[a1 - 1] - coords[a2 - 1]
        v1b = coords[a2 - 1] - coords[a3 - 1]
        n1 = np.cross(v1a, v1b)
        n1_l = np.linalg.norm(n1)

        v2a = coords[a4 - 1] - coords[a3 - 1]
        v2b = coords[a3 - 1] - coords[a2 - 1]
        n2 = np.cross(v2a, v2b)
        n2_l = np.linalg.norm(n2)

        ang = np.arccos(np.linalg.norm(np.dot(n1, n2)) / (n1_l * n2_l))
        ang_deg = ang * 180 / np.pi
        if corr_da:
            ang_deg = 180 - ang_deg
        if return_values:
            res.append(ang_deg)
        else:
            print(
                form_str_angle.format(
                    types[a1 - 1],
                    a1,
                    types[a2 - 1],
                    a2,
                    types[a3 - 1],
                    a3,
                    types[a4 - 1],
                    a4,
                    ang_deg,
                )
            )

    if return_values:
        return np.array(res)


def plane_angles(p_fn, coords, types, form_str_angle, autosave):
    with open(p_fn, "rt", encoding=enc) as f:
        lines = f.readlines()

    angles = []
    for s in lines:
        tmp = s.split(",")
        tmp = [int(t) for t in tmp]
        angles.append(tuple(tmp))

    print("# Plane angles", angles)

    for i, (p11, p12, p13, p21, p22, p23) in enumerate(angles):
        pv11 = coords[p11 - 1] - coords[p12 - 1]
        pv12 = coords[p12 - 1] - coords[p13 - 1]
        pv1 = np.cross(pv11, pv12)
        pv1n = np.linalg.norm(pv1)

        pv21 = coords[p21 - 1] - coords[p22 - 1]
        pv22 = coords[p22 - 1] - coords[p23 - 1]
        pv2 = np.cross(pv21, pv22)
        pv2n = np.linalg.norm(pv2)

        ang = np.arccos(np.linalg.norm(np.dot(pv1, pv2)) / (pv1n * pv2n))
        ang_deg = ang * 180 / np.pi
        print(
            form_str_angle.format(
                types[p11 - 1],
                p11,
                types[p12 - 1],
                p12,
                types[p13 - 1],
                p13,
                types[p21 - 1],
                p21,
                types[p22 - 1],
                p22,
                types[p23 - 1],
                p23,
                ang_deg,
            )
        )


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-xyz", required=True, help="File containing geometry")
    parser.add_argument("-o", help="File for customizing the order of atoms")
    parser.add_argument(
        "-b", help="File comtaiming atom pairs for which the bonds are analyzed"
    )
    parser.add_argument("-avgb", help="File comtaiming bonds to be averaged")
    parser.add_argument(
        "-a", help="File comtaiming atom triplets for which the angles are analyzed"
    )
    parser.add_argument("-avga", help="File contaiming angles to be averaged")
    parser.add_argument(
        "-t",
        help="File comtaiming atom quadruplets for which the dihedral angles are analyzed",
    )
    parser.add_argument("-d", type=str, default="3", help="Number of digits shown")
    parser.add_argument(
        "-da", type=str, default="1", help="Number of digits shown for angles"
    )
    parser.add_argument(
        "--angles", dest="angles", action="store_true", help="Dont calculate angles"
    )
    parser.add_argument(
        "--dihedrals",
        dest="dihedrals",
        action="store_true",
        help="Dont calculate dihedral angles",
    )
    parser.add_argument(
        "--autosave", action="store_true", help="Save default bonds and angles"
    )
    parser.add_argument(
        "--corr-da", action="store_true", help="Calculate dihedral as 180°-ang"
    )
    parser.add_argument(
        "--include-H",
        dest="inc_H",
        action="store_true",
        help="Include H atoms (only recommended for small linear molecules)",
    )
    parser.add_argument(
        "-p",
        help="""File contaiming two atom triplets defining the plane for
        which the angle should be calculated""",
    )
    args = parser.parse_args()

    print("# ", args)

    fn = args.xyz
    o_fn = args.o
    b_fn = args.b
    a_fn = args.a
    t_fn = args.t
    n_dig = args.d
    n_dig_a = args.da
    l_angles = args.angles
    l_dihedrals = args.dihedrals
    autosave = args.autosave
    inc_H = args.inc_H
    avgb = args.avgb
    avga = args.avga
    p_fn = args.p
    correct_da = args.corr_da

    form_str = "{}{}-{}{}\t:\t{:." + n_dig + "f}"
    form_str_angle = "{}{}-{}{}-{}{}\t:\t{:." + n_dig_a + "f}"
    form_str_dihedral = "{}{}-{}{}-{}{}-{}{}\t:\t{:." + n_dig_a + "f}"
    form_str_pa = "{}{}-{}{}-{}{}|{}{}-{}{}-{}{}\t:\t{:." + n_dig_a + "f}"
    form_str_avgb = "{}\t:\t{:." + n_dig + "f}"
    form_str_angle_avg = "{}\t:\t{:." + n_dig_a + "f}"

    coords = np.loadtxt(fn, skiprows=2, usecols=(1, 2, 3))
    types = np.loadtxt(fn, skiprows=2, usecols=(0), dtype=str)
    atomic_numbers = np.array(atoms.symbols2numbers(types))

    if not inc_H:
        idx = np.where(atomic_numbers > 1)[0]
        coords = coords[idx]
        types = types[idx]

    if o_fn is not None:
        sort_idx = (
            np.array([int(l) for i, l in enumerate(open(o_fn, "rt").readlines())]) - 1
        )
        coords = coords[sort_idx]
        types = types[sort_idx]

    bonds(b_fn, coords, types, form_str, autosave)

    if avgb is not None:
        avg_bonds(avgb, coords, types, form_str_avgb, autosave)

    if l_angles:
        angles(a_fn, coords, types, form_str_angle, autosave)

        if avga is not None:
            avg_angles(avga, coords, types, form_str_angle_avg, autosave)

    if l_dihedrals:
        dihedrals(t_fn, coords, types, form_str_dihedral, autosave, corr_da=correct_da)

    if p_fn is not None:
        plane_angles(p_fn, coords, types, form_str_pa, autosave)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
