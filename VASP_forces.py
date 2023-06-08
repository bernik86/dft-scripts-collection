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

from math import sqrt


def main():
    with open("vasprun.xml", "r") as f:
        lines = f.readlines()

    natoms = int(
        lines[[i for i, s in enumerate(lines) if "<atoms>" in s][0]].strip().split()[1]
    )
    print("Number of atoms", natoms)

    ind = [
        i
        for i, s in enumerate(lines)
        if ' <varray name="selective"  type="logical" >' in s
    ]

    constr = []
    if len(ind):
        ind = ind[0]
        for i in range(1, natoms + 1):
            if "F  F  F" in lines[ind + i]:
                constr.append(i)

    ind = [i for i, s in enumerate(lines) if 'name="forces" ' in s]

    for i in ind:
        force = 0.0
        at = 0
        for j in range(1, natoms + 1):
            if j in constr:
                continue
            temp = lines[i + j].strip().split()
            force2 = sqrt(
                float(temp[1]) ** 2 + float(temp[2]) ** 2 + float(temp[3]) ** 2
            )
            if force2 > force:
                force = force2
                at = j

        print("{0}\t{1:.5f}".format(at, force))

    return 0


if __name__ == "__main__":
    main()
