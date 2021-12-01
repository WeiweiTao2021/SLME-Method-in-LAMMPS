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
 
FixStyle(SABC,FixSABC)

#else

#ifndef LMP_FIX_SABC_H
#define LMP_FIX_SABC_H

#include "fix.h"

namespace LAMMPS_NS {
	
	class FixSABC : public Fix {
		friend class MinLineSearch;
		
	public:
		FixSABC(class LAMMPS *, int, char **);
		~FixSABC();
		int setmask();
        
        double *request_vector(int);
        void add_vector();
        void clear_vector();
		double memory_usage();
		void grow_arrays(int);
        
		void copy_arrays(int, int, int);
		int pack_exchange(int, double *);
		int unpack_exchange(int, double *);
		
		double *vstore;
        
	private:
        double **vectors;
		int nvector;
		
	};
	
}

#endif
#endif
