/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
 
#ifdef FIX_CLASS

FixStyle(abc,FixABC)

#else

#ifndef LMP_FIX_ABC_H
#define LMP_FIX_ABC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixABC : public Fix {
 public:
    FixABC(class LAMMPS *, int, char **);
    ~FixABC();
    int setmask();
    void init();
    void setup(int);
    void post_force(int);
    void pre_force(int);
    void min_setup(int);
    void set_Baddpf(int);
    void min_post_force(int);
    void min_pre_force(int);
    double compute_scalar();
    
    void copy_arrays(int, int, int);
    int pack_exchange(int, double *);
    int unpack_exchange(int, double *);
    
    void clear_vector();
    double enmax;
    
    double omega, wstart;
    int sigma;
    
 private:
    
    int store(int);
    void xxc(double *);
    void add_vector();
    void grow_arrays(int);
    void CombinePF(int);
    double force_energy(int);
    
    FILE *fp;
    int nvector;
    int store_no;
    double pert;
    double energy;
    int random_seed;
    double random_num;
    double dsaddle;
    bigint timestep;
    
    double *pwidth;
    double *pheight;
    double **vectors;
    
    class RanPark *random_unequal;
    class FixSABC *fix_sabc;
  
    int Bcombine;
  
    int Baddpf;
    int Blm;
    
    int maxstepflag;
    int maxstep;
};

}

#endif
#endif
