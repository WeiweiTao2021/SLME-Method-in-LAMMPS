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
 
#ifdef COMMAND_CLASS

CommandStyle(abc,ABC)

#else

#ifndef LMP_ABC_H
#define LMP_ABC_H

#include "pointers.h"

namespace LAMMPS_NS {

class ABC : protected Pointers {
    public:
        ABC(class LAMMPS *);
        void command(int, char **);
        void normalabc(int,char **);
        void min(int,char **);
    
    private:
        void storeabc();
        void xxc(double *);
        void remap();
        void write(int);
        void backtrack(int);
    
        class Minimize *minimize;
        class FixABC *fix_abc;
        class FixSABC *fix_sabc;
        class RanPark *random_mc;
        class Compute *presscom;
    
        FILE *fp;
        FILE *fp2;
    
        double ebarrier;
        double dmin;
        double ftolerance;
        double temperature;
        double maxstrain;
        int ndump;
        int nkeep;
        int lm_no;
        int steps;
        int maxpf;
        int seed_mc;
        int igroup,groupbit;
        int keep_lm;
        int store_lm;
    
        double compute_evdwl();
        double compute_ecoul();
        double compute_ebond();
        double compute_eangle();
        double compute_edihed();
        double compute_eimp();
        double compute_elong();
        double compute_density();
    
        double dilation[3];
};

}

#endif
#endif
