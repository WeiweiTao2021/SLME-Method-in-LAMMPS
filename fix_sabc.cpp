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
 
#include "stdlib.h"
#include "fix_sabc.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSABC::FixSABC(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
    
    // if (comm->me == 0) fprintf(screen,"IN FixSABC now!\n");
    
	nvector = 0;
	vectors = NULL;
    vstore = NULL;
	// register callback to this fix from Atom class
	// don't perform initial allocation here, must wait until add_vector()
	atom->add_callback(0);

}

/* ---------------------------------------------------------------------- */

FixSABC::~FixSABC()
{
	// unregister callbacks to this fix from Atom class
	
	atom->delete_callback(id,0);
    memory->destroy(vstore);
    
	for (int m = 0; m < nvector; m++) memory->destroy(vectors[m]);
	memory->sfree(vectors);

}

/* ---------------------------------------------------------------------- */

int FixSABC::setmask()
{
	return 0;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double FixSABC::memory_usage()
{
	double bytes = 0.0;
	for (int m = 0; m < nvector; m++)
		bytes += atom->nmax*3*sizeof(double);
	return bytes;
}


/* ----------------------------------------------------------------------
 allocate local atom-based arrays
 ------------------------------------------------------------------------- */

void FixSABC::grow_arrays(int nmax)
{
   // if (comm->me == 0) fprintf(screen,"FixSABC Grow_array\n");
    
    for (int m = 0; m < nvector; m++)
        memory->grow(vectors[m],3*nmax,"minimize:vector");
}

/* ----------------------------------------------------------------------
 Return vector
 ------------------------------------------------------------------------- */
double *FixSABC::request_vector(int m)
{
    //if (comm->me == 0) fprintf(screen,"FixSABC::request_vector m=%d nvector=%d\n",m, nvector);
    return vectors[m];
}

/* ----------------------------------------------------------------------
 copy values within local atom-based arrays
 ------------------------------------------------------------------------- */

void FixSABC::copy_arrays(int i, int j, int delflag)
{
    
    //if (comm->me == 0) fprintf(screen,"FixSABC copy_array!\n");
    if(nvector > 0){
        for(int ivector = 0; ivector < nvector; ivector++)
            for (int m = 0; m < 3; m++)
                vectors[ivector][3*j+m] = vectors[ivector][3*i+m];
    }
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for exchange with another proc
 ------------------------------------------------------------------------- */

int FixSABC::pack_exchange(int i, double *buf)
{
	//fprintf(screen,"FixSABC Pack_exchange nvector=%d i=%d\n", nvector,i);
	
    int n = 0;
    if(nvector > 0){
        for(int ivector = 0; ivector < nvector; ivector++)
            for (int m = 0; m < 3; m++)
                buf[n++] = vectors[ivector][3*i+m];
    }
    
    //fprintf(screen,"FixSABC Pack_exchange ends n =%d\n", n);
    
	return n;
    
    
}

/* ----------------------------------------------------------------------
 unpack values in local atom-based arrays from exchange with another proc
 ------------------------------------------------------------------------- */

int FixSABC::unpack_exchange(int nlocal, double *buf)
{
	//fprintf(screen,"FixSABC unpack_exchange nlocal=%d nvector=%d\n",nlocal, nvector);
	int n = 0;
    if(nvector > 0){
        for(int ivector = 0; ivector < nvector; ivector++)
            for (int m = 0; m < 3; m++)
                vectors[ivector][3*nlocal+m] = buf[n++];
    }
    
    //fprintf(screen,"FixSABC unpack_exchange ends n=%d\n",n);
	return n;
    
    
}

/* ----------------------------------------------------------------------
 allocate/initialize memory for a new vector with N elements per atom
 ------------------------------------------------------------------------- */

void FixSABC::add_vector()
{
    
    //if (comm->me == 0) fprintf(screen,"FixSABC ADD_vector nvector=%d!\n", nvector);
    
    memory->grow(vstore,3*nvector+3,"abc:pwidth");
    vstore[3*nvector+0] = 0.0;
    vstore[3*nvector+1] = 0.0;
    vstore[3*nvector+2] = 0.0;
    
    if(vectors == NULL) {
        memory->create(vectors,1,atom->nmax*3,"fixsabc:vectors");
    }
    else{
        vectors = (double **)
        memory->srealloc(vectors,(nvector+1)*sizeof(double *),"fixsabc:vectors");
        memory->create(vectors[nvector],atom->nmax*3,"fixsabc:vector");
        
    }
    
    int nlocal = atom->nlocal;
    int n = 0;
    for (int i = 0; i < nlocal; i++){
        vectors[nvector][n+0] = 0.0;
        vectors[nvector][n+1] = 0.0;
        vectors[nvector][n+2] = 0.0;
        n = n + 3;
    }
    
    nvector++;
    
}


/* ----------------------------------------------------------------------
 Clear vectors after each shear step
 ------------------------------------------------------------------------- */

void FixSABC::clear_vector()
{
    if (comm->me == 0) fprintf(screen,"FixSABC Clear Vector!\n");
    
    for (int m = 0; m < nvector; m++) memory->destroy(vectors[m]);
    memory->sfree(vectors);
    nvector = 0;
    vectors = NULL;
    
    memory->destroy(vstore);
    vstore = NULL;

}




