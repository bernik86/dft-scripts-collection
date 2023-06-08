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
""" VASP_get_axsf_from_vasprun.py reads the vasprun.xml file, extracts
the structures from it, and converts them to an animated xsf file
that can be opened by XCrysDen.
"""
from pymatgen.io import vasp, xcrysden


def main():
    vasprun = vasp.Vasprun(
        "vasprun.xml", parse_dos=False, parse_eigen=False, exception_on_bad_xml=False
    )

    axsf = [f"ANIMSTEPS {len(vasprun.structures)}"]
    for i, struct in enumerate(vasprun.structures):
        xsf = xcrysden.XSF(struct)
        xsf_lines = xsf.to_string().split("\n")
        xsf_lines[7] += f" {i + 1}"
        if i >= 1:
            xsf_lines = xsf_lines[7:]
        axsf.append("\n".join(xsf_lines))

    with open("structures.axsf", "wt", encoding="UTF-8") as axsf_file:
        axsf_file.write("\n".join(axsf))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
