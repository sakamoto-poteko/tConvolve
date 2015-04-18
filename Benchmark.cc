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

// Include own header file first
#include "Benchmark.h"

// System includes
#include <unistd.h>
#include <sys/times.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>
#include <cstring>
#include <cstdio>

#include <mkl.h>
#include <mpi.h>

#define _MM_HINT_T0     0
#define _MM_HINT_NT1    1
#define _MM_HINT_NT2    2
#define _MM_HINT_NTA    3

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define ALIGN64 __attribute__((aligned(64)))

Benchmark::Benchmark()
        : next(1)
{
}

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
int Benchmark::randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}

void Benchmark::init()
{
    // Initialize the data to be gridded
    u.resize(nSamples);
    v.resize(nSamples);
    w.resize(nSamples);
    samples.resize(nSamples*nChan);
    outdata.resize(nSamples*nChan);


    double rd;
    FILE * fp;
    if( (fp=fopen("randnum.dat","rb"))==NULL )
    {
      printf("cannot open file\n");
      return;
    }

    for (int i = 0; i < nSamples; i++) {
        if(fread(&rd,sizeof(double),1,fp)!=1){printf("Rand number read error!\n");}
        u[i] = baseline * rd - baseline / 2;
        if(fread(&rd,sizeof(double),1,fp)!=1){printf("Rand number read error!\n");}
        v[i] = baseline * rd - baseline / 2;
        if(fread(&rd,sizeof(double),1,fp)!=1){printf("Rand number read error!\n");}
        w[i] = baseline * rd - baseline / 2;

        for (int chan = 0; chan < nChan; chan++) {
            if(fread(&rd,sizeof(double),1,fp)!=1){printf("Rand number read error!\n");}
            samples[i*nChan+chan].data = rd;
            outdata[i*nChan+chan] = 0.0;
        }
    }
    fclose(fp);

    grid.resize(gSize*gSize);
    grid.assign(grid.size(), Value(0.0));

    // Measure frequency in inverse wavelengths
    std::vector<double> freq(nChan);

    for (int i = 0; i < nChan; i++) {
        freq[i] = (1.4e9 - 2.0e5 * double(i) / double(nChan)) / 2.998e8;
    }

    // Initialize convolution function and offsets
    initC(freq, cellSize, wSize, m_support, overSample, wCellSize, C);
    initCOffset(u, v, w, freq, cellSize, wCellSize, wSize, gSize,
                m_support, overSample);
}

void Benchmark::runGrid()
{
    gridKernel(m_support, C, grid, gSize);
}

/////////////////////////////////////////////////////////////////////////////////
// The next function is the kernel of the gridding.
// The data are presented as a vector. Offsets for the convolution function
// and for the grid location are precalculated so that the kernel does
// not need to know anything about world doubleinates or the shape of
// the convolution function. The ordering of cOffset and iu, iv is
// random.
//
// Perform gridding
//
// data - values to be gridded in a 1D vector
// support - Total width of convolution function=2*support+1
// C - convolution function shape: (2*support+1, 2*support+1, *)
// cOffset - offset into convolution function per data point
// iu, iv - integer locations of grid points
// grid - Output grid: shape (gSize, *)
// gSize - size of one axis of grid

void Benchmark::gridKernel(const int support,
                           const std::vector<Value>& C,
                           std::vector<Value>& grid, const int gSize)
{
    // Constant: sSize, support, gSize
    const int sSize = 2 * support + 1;
    const int grid_ary_byte_size = grid.size() * 2 * sizeof(double);
    const int avail_mics = mkl_mic_get_device_count();
    const int sampleSize = samples.size();
    const float host_mic_work_division = 0.45;
    int proc_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const int samples_per_proc = sampleSize / world_size;
    const int rank0_sample_size = sampleSize % world_size + samples_per_proc;
    int MPI_start_dind              = proc_rank ? (proc_rank - 1) * samples_per_proc + rank0_sample_size : 0;
    int MPI_allocated_sample_size   = proc_rank ? samples_per_proc : rank0_sample_size;

#ifdef VERBOSE
    if (!proc_rank) {
        printf("\nMPI Report:\n"
               "\tTotal Processes:%8d\n"
               "\tTotal Samples:  %8d\n"
               "\tSamples/Process:%8d\n"
               "\tRoot Samples:   %8d\n",
               world_size, sampleSize, samples_per_proc, rank0_sample_size);
    }
#endif

    assert(gSize > sSize);

    const double * __restrict__ samples_ary = this->samples_ary;
    const double * __restrict__ C_ary       = this->C_ary;
    double * __restrict__ grid_ary          = this->grid_ary;
    double  per_mic_out_grid[avail_mics][grid.size() * 2]   ALIGN64;

    __assume_aligned(samples_ary, 64);
    __assume_aligned(C_ary, 64);
    __assume_aligned(grid_ary, 64);

    double cinds[sampleSize]    ALIGN64;
    double ginds[sampleSize]    ALIGN64;
    double dreals[sampleSize]   ALIGN64;
    double dimags[sampleSize]   ALIGN64;
    for (int i = 0; i < sampleSize; ++i) {
        cinds[i] = samples[i].cOffset;
        ginds[i] = samples[i].iu + gSize * samples[i].iv - support;
        dreals[i] = samples_ary[i * 2];
        dimags[i] = samples_ary[i * 2 + 1];
    }

    int dev_work_loads[avail_mics + 1]; // + 1 for host, this is the start dind for devices

    int host_total_samples = MPI_allocated_sample_size * host_mic_work_division;
    const int mic_total_samples = MPI_allocated_sample_size - host_total_samples;
    const int samples_per_mic = mic_total_samples / avail_mics;   // There's a remainder. Add to host
    host_total_samples += mic_total_samples % avail_mics;

    for (int i = 0; i < avail_mics; ++i) {
        dev_work_loads[i] = samples_per_mic * i + MPI_start_dind;
    }
    dev_work_loads[avail_mics] = samples_per_mic * avail_mics + MPI_start_dind;



    /// Host: Schedule work for MICs
    for (int imic = 0; imic < avail_mics; ++imic) {
        const int start_dind = dev_work_loads[imic];
        const int end_dind = dev_work_loads[imic + 1];

        double * __restrict__ my_out_grid_ary = per_mic_out_grid[imic];

        char memset_comp_sgnl;
        #pragma offload target(mic:imic) in(C_ary:length(C.size() * 2) align(64) ALLOC) \
                                         in(cinds:length(sampleSize) align(64) ALLOC) \
                                         in(ginds:length(sampleSize) align(64) ALLOC) \
                                         in(dreals:length(sampleSize) align(64) ALLOC) \
                                         in(dimags:length(sampleSize) align(64) ALLOC) \
                                         nocopy(my_out_grid_ary:length(grid.size() * 2) align(64) ALLOC) \
                                         out(memset_comp_sgnl) in(grid_ary_byte_size) \
                                         signal(&memset_comp_sgnl)
        {
            memset(my_out_grid_ary, 0, grid_ary_byte_size);
            memset_comp_sgnl = 0;
        }

        #pragma offload target(mic:imic) in(sampleSize) in(sSize) in(gSize) \
                                         in(start_dind) in(end_dind) \
                                         out(my_out_grid_ary:length(grid.size() * 2) align(64) FREE) \
                                         nocopy(cinds:length(sampleSize) align(64) FREE) \
                                         nocopy(ginds:length(sampleSize) align(64) FREE) \
                                         nocopy(dreals:length(sampleSize) align(64) FREE) \
                                         nocopy(dimags:length(sampleSize) align(64) FREE) \
                                         nocopy(C_ary:length(C.size() * 2) align(64) FREE) \
                                         signal(my_out_grid_ary) wait(&memset_comp_sgnl)
        {
            offloadKernel(C_ary, dreals, ginds, gSize, start_dind, sSize, end_dind, my_out_grid_ary, dimags, cinds);
        }
    }

#ifdef VERBOSE
    struct tms t;
    clock_t start_t = times(&t);
    if (!proc_rank) {
        printf("MIC Offload Kernels Started\n"
               "Per Node Report:\n"
               "\tTotal MICs:             %6d\n"
               "\tHost/MIC Load Division: %2.0f%%\n"
               "\tTotal Samples for MICs: %6d\n"
               "\tSamples/MIC:            %6d\n"
               "\tSamples/Host:           %6d\n",
               avail_mics, host_mic_work_division * 100,
               mic_total_samples, samples_per_mic, host_total_samples);
    }
#endif

    /// Host begins calculation. Parallel with MICs
    offloadKernel(C_ary, dreals, ginds, gSize, dev_work_loads[avail_mics], sSize,
                  dev_work_loads[avail_mics] + host_total_samples, grid_ary, dimags, cinds);
    /// Host completed calculation.

    /// Host waits for MIC completion, then reduce.
    #pragma omp parallel for schedule(static)
    for (int imic = 0; imic < avail_mics; ++imic) {
        #pragma offload_wait target(mic:imic) wait(per_mic_out_grid[imic])

        #pragma omp critical
        {
            vdAdd(grid.size() * 2, grid_ary, per_mic_out_grid[imic], grid_ary);
        }
    }

    if (!proc_rank) {
#ifdef VERBOSE
        printf("Reducing ...\n");
#endif
        MPI_Reduce(MPI_IN_PLACE, grid_ary, grid.size() * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#ifdef VERBOSE
        clock_t end_t = times(&t);
        const double elapsed = static_cast<double>(end_t - start_t) / static_cast<double>(sysconf(_SC_CLK_TCK));
        const double griddings = (double(nSamples * nChan) * double((sSize) * (sSize)));
        printf("Reduced\n"
               "Actual Computation Time (w/o tsfr & print call): %4.4fs, "
               "i.e. %4.4f billions of grid points per second\n",
               elapsed, griddings / 1000000000. / elapsed);
#endif
    } else {
        MPI_Reduce(grid_ary, grid_ary, grid.size() * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

__attribute__((target(mic))) void Benchmark::offloadKernel(const double* __restrict__ C_ary,
                                                           const double *dreals,
                                                           const double *ginds,
                                                           const int gSize,
                                                           const int start_dind,
                                                           const int sSize,
                                                           const int end_dind,
                                                           double* __restrict__ grid_ary,
                                                           const double *dimags,
                                                           const double *cinds)
{
    for (int dind = start_dind; dind < end_dind; ++dind) {
        const double d_real = dreals[dind];
        const double d_imag = dimags[dind];
        const int gind = ginds[dind];
        const int cind = cinds[dind];

        #pragma omp parallel for simd
        #pragma ivdep
        #pragma prefetch C_ary:_MM_HINT_NT1:64
        #pragma prefetch grid_ary:_MM_HINT_NTA:64
        #pragma vector nontemporal (grid_ary)
        for (int i = 0; i < sSize * sSize; ++i) {
                int suppv = i / sSize;
                int suppu = i % sSize;

                const double c_real = C_ary[2 * (cind + sSize * suppv + suppu)    ];
                const double c_imag = C_ary[2 * (cind + sSize * suppv + suppu) + 1];

                const double calc_greal = d_real * c_real - d_imag * c_imag;
                const double calc_gimag = d_real * c_imag + d_imag * c_real;

                grid_ary[2 * (gind + gSize * suppv + suppu)    ] += calc_greal;
                grid_ary[2 * (gind + gSize * suppv + suppu) + 1] += calc_gimag;
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// wSize - Size of lookup table in w
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
void Benchmark::initC(const std::vector<double>& freq,
                      const double cellSize, const int wSize,
                      int& support, int& overSample,
                      double& wCellSize, std::vector<Value>& C)
{
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<double>(cellSize)
                                          * freq[0]) / cellSize);

    overSample = 8;
    wCellSize = 2 * baseline * freq[0] / wSize;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    const int sSize = 2 * support + 1;

    const int cCenter = (sSize - 1) / 2;

    C.resize(sSize*sSize*overSample*overSample*wSize);

    for (int k = 0; k < wSize; k++) {
        double w = double(k - wSize / 2);
        double fScale = sqrt(std::abs(w) * wCellSize * freq[0]) / cellSize;

        for (int osj = 0; osj < overSample; osj++) {
            for (int osi = 0; osi < overSample; osi++) {
                for (int j = 0; j < sSize; j++) {
                    double j2 = std::pow((double(j - cCenter) + double(osj) / double(overSample)), 2);

                    for (int i = 0; i < sSize; i++) {
                        double r2 = j2 + std::pow((double(i - cCenter) + double(osi) / double(overSample)), 2);
                        long int cind = i + sSize * (j + sSize * (osi + overSample * (osj + overSample * k)));

                        if (w != 0.0) {
                            C[cind] = static_cast<Value>(std::cos(r2 / (w * fScale)));
                        } else {
                            C[cind] = static_cast<Value>(std::exp(-r2));
                        }
                    }
                }
            }
        }
    }

    // Now normalise the convolution function
    double sumC = 0.0;

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        sumC += std::abs(C[i]);
    }

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        C[i] *= Value(wSize * overSample * overSample / sumC);
    }

    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    if (!proc_rank) {
        std::cout << "Initializing W projection convolution function" << std::endl;
        std::cout << "Support = " << support << " pixels" << std::endl;
        std::cout << "W cellsize = " << wCellSize << " wavelengths" << std::endl;
        std::cout << "Size of convolution function = " << sSize*sSize*overSample
                  *overSample*wSize*sizeof(Value) / (1024*1024) << " MB" << std::endl;
        std::cout << "Shape of convolution function = [" << sSize << ", " << sSize << ", "
                  << overSample << ", " << overSample << ", " << wSize << "]" << std::endl;

    }
}

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void Benchmark::initCOffset(const std::vector<double>& u, const std::vector<double>& v,
                            const std::vector<double>& w, const std::vector<double>& freq,
                            const double cellSize, const double wCellSize,
                            const int wSize, const int gSize, const int support,
                            const int overSample)
{
    const int nSamples = u.size();
    const int nChan = freq.size();

    const int sSize = 2 * support + 1;

    // Now calculate the offset for each visibility point
    for (int i = 0; i < nSamples; i++) {
        for (int chan = 0; chan < nChan; chan++) {

            int dind = i * nChan + chan;

            double uScaled = freq[chan] * u[i] / cellSize;
            samples[dind].iu = int(uScaled);

            if (uScaled < double(samples[dind].iu)) {
                samples[dind].iu -= 1;
            }

            int fracu = int(overSample * (uScaled - double(samples[dind].iu)));
            samples[dind].iu += gSize / 2;

            double vScaled = freq[chan] * v[i] / cellSize;
            samples[dind].iv = int(vScaled);

            if (vScaled < double(samples[dind].iv)) {
                samples[dind].iv -= 1;
            }

            int fracv = int(overSample * (vScaled - double(samples[dind].iv)));
            samples[dind].iv += gSize / 2;

            // The beginning of the convolution function for this point
            double wScaled = freq[chan] * w[i] / wCellSize;
            int woff = wSize / 2 + int(wScaled);
            samples[dind].cOffset = sSize * sSize * (fracu + overSample * (fracv + overSample * woff));
        }
    }
}

void Benchmark::printGrid()
{
  FILE * fp;
  if( (fp=fopen("grid.dat","wb"))==NULL )
  {
    printf("cannot open file\n");
    return;
  }

  unsigned ij;
  for (int i = 0; i < gSize; i++)
  {
    for (int j = 0; j < gSize; j++)
    {
      ij=j+i*gSize;
      if(fwrite(&grid[ij],sizeof(Value),1,fp)!=1)
        printf("File write error!\n");

    }
  }

  fclose(fp);
}

int Benchmark::getSupport()
{
    return m_support;
};
