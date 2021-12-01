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

#include <mpi.h>
#include <math.h>
#include "min_quickmin.h"
#include "universe.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "output.h"
#include "timer.h"
#include "error.h"
#include "comm.h"

//ADD BY WEIWEI
#include "domain.h"
#include "group.h"

using namespace LAMMPS_NS;

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8

// same as in other min classes

enum{MAXITER,MAXEVAL,ETOL,FTOL,DOWNHILL,ZEROALPHA,ZEROFORCE,ZEROQUAD};

#define DELAYSTEP 5

/* ---------------------------------------------------------------------- */

MinQuickMin::MinQuickMin(LAMMPS *lmp) : Min(lmp) {}

/* ---------------------------------------------------------------------- */

void MinQuickMin::init()
{
  Min::init();

  dt = update->dt;
  last_negative = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void MinQuickMin::setup_style()
{
  double **v = atom->v;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    v[i][0] = v[i][1] = v[i][2] = 0.0;
}

/* ----------------------------------------------------------------------
   set current vector lengths and pointers
   called after atoms have migrated
------------------------------------------------------------------------- */

void MinQuickMin::reset_vectors()
{
  // atomic dof

  nvec = 3 * atom->nlocal;
  if (nvec) xvec = atom->x[0];
  if (nvec) fvec = atom->f[0];
}

/* ----------------------------------------------------------------------
   minimization via QuickMin damped dynamics
------------------------------------------------------------------------- */

int MinQuickMin::iterate(int maxiter)
{
  bigint ntimestep;
  double vmax,vdotf,vdotfall,fdotf,fdotfall,scale;
  double dtvone,dtv,dtf,dtfm;
  int flag,flagall;

  alpha_final = 0.0;

  for (int iter = 0; iter < maxiter; iter++) {

    ntimestep = ++update->ntimestep;
    niter++;

    // zero velocity if anti-parallel to force
    // else project velocity in direction of force

    double **v = atom->v;
    double **f = atom->f;
    int nlocal = atom->nlocal;

    vdotf = 0.0;
    for (int i = 0; i < nlocal; i++)
      vdotf += v[i][0]*f[i][0] + v[i][1]*f[i][1] + v[i][2]*f[i][2];
    MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,world);

    // sum vdotf over replicas, if necessary
    // this communicator would be invalid for multiprocess replicas

    if (update->multireplica == 1) {
      vdotf = vdotfall;
      MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
    }

    if (vdotfall < 0.0) {
      last_negative = ntimestep;
      for (int i = 0; i < nlocal; i++)
        v[i][0] = v[i][1] = v[i][2] = 0.0;

    } else {
      fdotf = 0.0;
      for (int i = 0; i < nlocal; i++)
        fdotf += f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2];
      MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,world);

      // sum fdotf over replicas, if necessary
      // this communicator would be invalid for multiprocess replicas

      if (update->multireplica == 1) {
        fdotf = fdotfall;
        MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }

      if (fdotfall == 0.0) scale = 0.0;
      else scale = vdotfall/fdotfall;
      for (int i = 0; i < nlocal; i++) {
        v[i][0] = scale * f[i][0];
        v[i][1] = scale * f[i][1];
        v[i][2] = scale * f[i][2];
      }
    }

    // limit timestep so no particle moves further than dmax

    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;

    dtvone = dt;

    for (int i = 0; i < nlocal; i++) {
      vmax = MAX(fabs(v[i][0]),fabs(v[i][1]));
      vmax = MAX(vmax,fabs(v[i][2]));
      if (dtvone*vmax > dmax) dtvone = dmax/vmax;
    }
    MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,world);

    // min dtv over replicas, if necessary
    // this communicator would be invalid for multiprocess replicas

    if (update->multireplica == 1) {
      dtvone = dtv;
      MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,universe->uworld);
    }

    dtf = dtv * force->ftm2v;

      
      //ADD BY WEIWEI (REMOVE THE AGULAR MOMENTUM)
      int igroup;
      igroup = group->find("all");
      double masstotal = group->mass(igroup);
      double xcm[3],angmom[3],inertia[3][3],omega[3];
      group->xcm(igroup,masstotal,xcm);
      group->angmom(igroup,xcm,angmom);
      group->inertia(igroup,xcm,inertia);
      group->omega(angmom,inertia,omega);
      
       //if(iter % 50 == 0) fprintf(screen,"omega %f %f %f !\n",omega[0], omega[1], omega[2]);
      
      // adjust velocities to zero omega
      // vnew_i = v_i - w x r_i
      // must use unwrapped coords to compute r_i correctly
      
      double **x = atom->x;
      int *mask = atom->mask;
      imageint *image = atom->image;
      
      double dx,dy,dz;
      double unwrap[3];
      
      for (int i = 0; i < nlocal; i++){
            domain->unmap(x[i],image[i],unwrap);
            dx = unwrap[0] - xcm[0];
            dy = unwrap[1] - xcm[1];
            dz = unwrap[2] - xcm[2];
            v[i][0] -= omega[1]*dz - omega[2]*dy;
            v[i][1] -= omega[2]*dx - omega[0]*dz;
            v[i][2] -= omega[0]*dy - omega[1]*dx;
      }
      
      
      
    // Euler integration step
    if (rmass) {
      for (int i = 0; i < nlocal; i++) {
        dtfm = dtf / rmass[i];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
          //if(iter % 50 == 0) fprintf(screen,"v[i][0] %f %f %f !\n",v[i][0], v[i][1], v[i][2]);
          
      }
    } else {
      for (int i = 0; i < nlocal; i++) {
        dtfm = dtf / mass[type[i]];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
          
           //if(iter % 50 == 0) fprintf(screen,"v[i][0] %f %f %f !\n",v[i][0], v[i][1], v[i][2]);
          
      }
    }

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    // energy tolerance criterion
    // only check after DELAYSTEP elapsed since velocties reset to 0
    // sync across replicas if running multi-replica minimization

    if (update->etol > 0.0 && ntimestep-last_negative > DELAYSTEP) {
      if (update->multireplica == 0) {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY) && iter >20)
          return ETOL;
      } else {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return ETOL;
      }
    }

    // force tolerance criterion
    // sync across replicas if running multi-replica minimization

    if (update->ftol > 0.0) {
        
      fdotf = fnorm_sqr();
        
      //if( comm->me == 0 && iter%50 == 0) fprintf(screen,"iter =%d fdotf =  %15.10g fdotf/natoms =  %15.10g update->ftol =  %15.10g!\n",iter, fdotf, fdotf/atom->natoms, update->ftol);
        
        if (update->multireplica == 0) {
        if (fdotf < update->ftol*update->ftol && iter >50) return FTOL;
      } else {
        if (fdotf < update->ftol*update->ftol && iter >50) flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return FTOL;
      }
    }

    // output for thermo, dump, restart files

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}
