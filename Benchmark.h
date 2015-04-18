/// @copyright (c) 2007 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
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

#ifndef BENCHMARK_H
#define BENCHMARK_H

// System includes
#include <vector>
#include <complex>
#include <stdio.h>

// Typedefs
typedef std::complex<double> Value;

struct Sample {
    Value data;
    int iu;
    int iv;
    int cOffset;
};

class Benchmark {
    public:
        Benchmark();

        int randomInt();
        void init();
        void runGrid();
        void runDegrid();

        void gridKernel(const int support,
                        const std::vector<Value>& C,
                        std::vector<Value>& grid, const int gSize);

        void initC(const std::vector<double>& freq,
                   const double cellSize, const int wSize,
                   int& support, int& overSample,
                   double& wCellSize, std::vector<Value>& C);

        void initCOffset(const std::vector<double>& u, const std::vector<double>& v,
                         const std::vector<double>& w, const std::vector<double>& freq,
                         const double cellSize, const double wCellSize, const int wSize,
                         const int gSize, const int support, const int overSample);

        int getSupport();

        void printGrid();


        void prepareData();
        void postProcessData();

// Change these if necessary to adjust run time
        int nSamples; // Number of data samples
        int wSize; // Number of lookup planes in w projection
        int nChan; // Number of spectral channels

// Don't change any of these numbers unless you know what you are doing!
        int gSize; // Size of output grid in pixels
        double cellSize; // Cellsize of output grid in wavelengths
        int baseline; // Maximum baseline in meters

        __attribute__((target (mic))) void offloadKernel(const double* __restrict__ C_ary,
                                                           const double * __restrict__ dreals,
                                                           const double * __restrict__ ginds,
                                                           const int gSize,
                                                           const int start_dind,
                                                           const int sSize,
                                                           const int end_dind,
                                                           double* __restrict__ grid_ary,
                                                           const double * __restrict__ dimags,
                                                           const double * __restrict__ cinds);
    private:
        std::vector<Value> grid;
        std::vector<double> u;
        std::vector<double> v;
        std::vector<double> w;
        std::vector<Sample> samples;
        std::vector<Value> outdata;

        std::vector< Value > C;
        int m_support;
        int overSample;

        double wCellSize;

        // For random number generator
        unsigned long next;

        double *samples_ary;
        double *C_ary;
        double *grid_ary;
};
#endif
