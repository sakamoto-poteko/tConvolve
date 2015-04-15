/*
/// @copyright (c) 2007 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia]
/// (c) 2015 Afa.L Cheng <afa@afa.moe>
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// This program was modified so as to use it in the contest.
/// The last modification was on January 12, 2015.
///
*/

// Include own header file first
#include "Benchmark.h"

// System includes
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#include <string.h>
#include <stdlib.h>

void Benchmark::prepareData()
{
    int samples_len = samples.size();
    int C_len = C.size();
    int grid_len = grid.size();

    samples_ary = (double *)_mm_malloc(samples_len * 2 * sizeof(double),  64);
    C_ary       = (double *)_mm_malloc(C_len * 2 * sizeof(double),        64);
    grid_ary    = (double *)_mm_malloc(grid_len * 2 * sizeof(double),     64);
    memset(grid_ary, 0, grid_len * 2 * sizeof(double));

    for (int i = 0; i < samples_len; ++i) {
        samples_ary[i * 2] = samples.at(i).data.real();
        samples_ary[i * 2 + 1] = samples.at(i).data.imag();
    }
    for (int i = 0; i < C_len; ++i) {
        C_ary[i * 2] = C.at(i).real();
        C_ary[i * 2 + 1] = C.at(i).imag();
    }
}

void Benchmark::postProcessData()
{
    for (int i = 0; i < grid.size(); ++i) {
        grid[i] = Value(grid_ary[2 * i], grid_ary[2 * i + 1]);
    }
}
