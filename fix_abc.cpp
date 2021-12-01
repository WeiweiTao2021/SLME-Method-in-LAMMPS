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

#include "fix_abc.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "mpi.h"
#include "math.h"
#include "error.h"
#include "stdlib.h"
#include "string.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
#include "fix.h"
#include "comm.h"
#include "random_park.h"
#include "abc.h"
#include "fix_sabc.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixABC::FixABC(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
    //from fix.h
    scalar_flag = 1;            // 0/1 if compute_scalar() function exists
    vector_flag = 1;
    global_freq = 1;            // frequency s/v data is available
    extscalar = 1;              // 0/1 if global scalar is intensive/extensive
    thermo_energy = 1;          // 1 if fix_modify enabled ThEng, 0 if not
    
    
    omega = atof(arg[3]);
    sigma = atoi(arg[4]);
    wstart = atof(arg[5]);
    random_seed =atoi(arg[6]);
    random_unequal = new RanPark(lmp,random_seed);
    Bcombine = atoi(arg[7]);
    pert = atof(arg[8]);
    
    maxstepflag = 0;
    if (narg == 10){
        maxstep = atoi(arg[11]);
        maxstepflag = 1;
    }
    
    enmax = -100000000;
    Baddpf = 1;                //whether add new PFs during minimization

    
    energy = 0.0;
    store_no = 0;
    nvector = 0;
    vectors = NULL;
    pwidth = NULL;
    pheight = NULL;
    
    //register callback to this fix from Atom class so it can manage atom-based arrays happens when fix is created; flag = 0 for grow;
    //don't perform initial allocation here, must wait until add_vector()
    atom->add_callback(0);
    
}

/* ---------------------------------------------------------------------- */
FixABC::~FixABC()
{
    
    for (int m = 0; m < nvector; m++) memory->destroy(vectors[m]);
    memory->sfree(vectors);
    
    memory->destroy(pwidth);
    memory->destroy(pheight);
    
    atom->delete_callback(id,0);
    
   //if (fp && comm->me == 0) fclose(fp);
    
}
/* ---------------------------------------------------------------------- */

int FixABC::setmask()
{
	int mask = 0;
	mask |= THERMO_ENERGY;
	mask |= MIN_PRE_FORCE;
	mask |= MIN_POST_FORCE;
	mask |= PRE_FORCE;
	mask |= POST_FORCE;
    
	return mask;
}


/* ----------------------------------------------------------------------
 Baddpf determines whether nedd to add new PF
 ------------------------------------------------------------------------- */

void FixABC::set_Baddpf(int add)
{
    Baddpf = add;
}

/* ----------------------------------------------------------------------
 allocate/initialize memory for a new vector with N elements per atom
 ------------------------------------------------------------------------- */

void FixABC::add_vector()
{
    
    //to save the width and height of the penalty functions
    memory->grow(pwidth,nvector+1,"abc:pwidth");
    memory->grow(pheight,nvector+1,"abc:pheight");

    
    //to save the center of the penalty functions
    if(vectors == NULL) {
        memory->create(vectors,1,atom->nmax*3,"fixsabc:vectors");
    }else{
        vectors = (double **) memory->srealloc(vectors,(nvector+1)*sizeof(double *),"abc:vectors");
        memory->create(vectors[nvector],atom->nmax*3,"abc:vectors");
    }
    
    //initialize the vectors
    int nlocal = atom->nlocal;
    int n = 0;
    for (int i = 0; i < nlocal; i++){
        vectors[nvector][n+0] = 0.0;
        vectors[nvector][n+1] = 0.0;
        vectors[nvector][n+2] = 0.0;
        n = n + 3;
    }
    
    store_no = store_no + 1;
    
    //if(comm->me == 0) fprintf(screen,"FixABC Add_vector comm->me=%d nmax=%d nvector = %d store_no=%d nlocal=%d\n",comm->me, atom->nmax,nvector, store_no,  atom->nlocal);

}

/* ----------------------------------------------------------------------*/

void FixABC::clear_vector()
{
    if (comm->me == 0) fprintf(screen,"FixABC clear_vector!\n");
    
    for (int m = 0; m < nvector; m++) memory->destroy(vectors[m]);
    memory->sfree(vectors);
    
    memory->destroy(pwidth);
    memory->destroy(pheight);
    
    store_no = 0;
    nvector = 0;
    vectors = NULL;
    pwidth = NULL;
    pheight = NULL;
    
    enmax = -100000000;
    
    random_unequal = new RanPark(lmp,random_seed);
    
}


/* ----------------------------------------------------------------------
 allocate local atom-based arrays
 ------------------------------------------------------------------------- */

void FixABC::grow_arrays(int nmax)
{
    for (int m = 0; m < nvector; m++)
		memory->grow(vectors[m],3*nmax,"abc:vectors");
}

/* ----------------------------------------------------------------------
 copy values within local atom-based arrays
 ------------------------------------------------------------------------- */

void FixABC::copy_arrays(int i, int j, int delflag)
{
    
    //fprintf(screen,"FixABC copy_array!\n");
    if(nvector > 0){
        for(int ivector = 0; ivector < nvector; ivector++)
            for (int m = 0; m < 3; m++)
                vectors[ivector][3*j+m] = vectors[ivector][3*i+m];
    }
    
   
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for exchange with another proc
 ------------------------------------------------------------------------- */

int FixABC::pack_exchange(int i, double *buf)
{
    
	
    //fprintf(screen,"FixABC Pack_exchange nvector=%d i=%d\n", nvector,i);
    
    int n = 0;
    if(nvector > 0){
        for(int ivector = 0; ivector < nvector; ivector++)
            for (int m = 0; m < 3; m++)
                buf[n++] = vectors[ivector][3*i+m];
    }
    
    //fprintf(screen,"FixABC Pack_exchange ends n=%d\n", n);
    return n;
    
    
    
}

/* ----------------------------------------------------------------------
 unpack values in local atom-based arrays from exchange with another proc
 ------------------------------------------------------------------------- */

int FixABC::unpack_exchange(int nlocal, double *buf)
{
  
    //fprintf(screen,"FixABC unpack_exchange nlocal=%d\n", nlocal);
    
	int n = 0;
    if(nvector > 0){
        for(int ivector = 0; ivector < nvector; ivector++)
            for (int m = 0; m < 3; m++)
                vectors[ivector][3*nlocal+m] = buf[n++];
    }
    //fprintf(screen,"FixABC unpack_exchange ends n=%d\n",n);
   	return n;
}

/*------------------------------------------------------------------------- */
void FixABC::init()
{
    
}
/*------------------------------------------------------------------------- */
void FixABC::setup(int vflag)
{
    //FOR DEBUG
    if( comm->me == 0)fprintf(screen,"FixABC::setup\n");
    
    int step_no_temp = nvector;
    timestep = update->ntimestep;
}
/* ---------------------------------------------------------------------- */
void FixABC::pre_force(int vflag)
{
	min_pre_force(vflag);
}
/* ---------------------------------------------------------------------- */
void FixABC::post_force(int vflag)
{
   	min_post_force(vflag);
}
/* ---------------------------------------------------------------------- */
void FixABC::min_setup(int vflag)
{
    //FOR DEBUG
    //if( comm->me == 0)fprintf(screen,"FixABC::min_setup\n");
    //if (comm->me == 0) fprintf(screen,"FIXABC omega=%f sigma=%d wstart=%f randpf=%d\n", omega, sigma, wstart, random_seed);
    
    
	int step_no_temp = nvector;
    
    if(Baddpf){
        store(step_no_temp);
        
         if(nvector >= 2 && Bcombine) {
                CombinePF(nvector-1);
         }
    }
    random_num = 2*random_unequal->uniform() - 1.0;
    
    //if (comm->me == 0) fprintf(screen,"MINsetup Random_num = %f nlocal = %d \n", random_num, atom->nlocal);
    
	min_post_force(1);
	timestep = update->ntimestep;
}
/* ---------------------------------------------------------------------- */
void FixABC::min_pre_force(int vflag)
{
	if (update->ntimestep != timestep) {
		timestep = update->ntimestep;
	}
}
/* ---------------------------------------------------------------------- */
void FixABC::min_post_force(int vflag)
{
    
    //ADD PFs
    int step_no_temp = nvector;
    energy = 0.0;
    double pf_energy = 0.0;
    
    if(Baddpf){
        for (int i = 0; i < step_no_temp; i++) {
            energy = force_energy(i);
            pf_energy += energy;
        }
   
    
        //Calculate f_org (without PFs after minimization)
        double **f = atom->f;
        int nlocal = atom->nlocal;
        double f_org = 0.0;
        for(int i = 0; i < nlocal; i++){
            for(int s = 0; s < 3; s++){
                f_org = f_org + f[i][s]*f[i][s];
            }
        }
        double f_all = 0.0;
        int natoms = atom->natoms;
        MPI_Allreduce(&f_org,&f_all,1,MPI_DOUBLE,MPI_SUM,world);
        f_all = f_all/natoms;
    
    
        int id = modify->find_compute((char *)"10");
        if (id < 0)   error->all(FLERR,"abc_pe ID for abc_pe does not exist");
        double current_energy = modify->compute[id]->scalar;
    
    
    
        //IF IT IS SADDLE POINT
        if( f_all < 0.0005 &&  current_energy > enmax){
            enmax = current_energy;
        }
    

        //FOR DEBUG
        if( update->ntimestep % 200 == 0 ){
            if(comm->me == 0) fprintf(screen,"Current_energy = %f f_all = %f pf_energy = %f step_no_temp = %d\n",current_energy, f_all, pf_energy, step_no_temp);
            
            //fprintf(screen," comm->me =%d nlocal=%d\n", comm->me, nlocal);
        }
    }
    
    
    
}


/* ---------------------------------------------------------------------- */
double FixABC::compute_scalar()
{
	return energy;
}
/* -----------------------------ADD PENALTY FUNCTIONs---------------------------------------- */
double FixABC::force_energy(int step)
{
    //CURRENT COORDINATES
    int nlocal = atom->nlocal;
	double **x = atom->x;
	double **f = atom->f;
	int *image = atom->image;
	double y[3];
    double xc[3];
    xxc(xc);
    
    //Center of PF
    double *x0;
    double width, height;
    x0 = vectors[step];
    width = pwidth[step];
    height = pheight[step];
    
    int *mask = atom->mask;
    double dx,dy,dz;
    double *h = domain->h;
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;
    int xbox,ybox,zbox;
    
    //calculate the distance between now and the center of pfs
	double dxdotdxme = 0.0;
    if(domain->triclinic == 0){
        int n = 0;
        for (int i = 0; i < nlocal; i++){
            if (mask[i] & groupbit) {
                
                //unwrap the coordinates to compare the real distance
                xbox = (image[i] & IMGMASK) - IMGMAX;
                ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
                zbox = (image[i] >> IMG2BITS) - IMGMAX;
            
                dx = x[i][0] + xbox*xprd - xc[0] - x0[n];
                dy = x[i][1] + ybox*yprd - xc[1] - x0[n+1];
                dz = x[i][2] + zbox*zprd - xc[2] - x0[n+2];
                
                //add a perturbations to current coordinate
                if(dx <= 1e-12) dx = dx + random_num*pert;
                if(dy <= 1e-12) dy = dy + random_num*pert;
                if(dz <= 1e-12) dz = dz + random_num*pert;
    
                dxdotdxme += dx*dx+ dy*dy+ dz*dz;
            }
            n += 3;
        }
    }else{
        int n = 0;
        for (int i = 0; i < nlocal; i++){
            if (mask[i] & groupbit) {
                
                xbox = (image[i] & IMGMASK) - IMGMAX;
                ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
                zbox = (image[i] >> IMG2BITS) - IMGMAX;
                
                dx = x[i][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox - xc[0] - x0[n];
                dy = x[i][1] + h[1]*ybox + h[3]*zbox - xc[1] - x0[n+1];
                dz = x[i][2] + h[2]*zbox - xc[2] - x0[n+2];
                
                
                if(dx <= 1e-12) dx = dx + random_num*pert;
                if(dy <= 1e-12) dy = dy + random_num*pert;
                if(dz <= 1e-12) dz = dz + random_num*pert;
                
                dxdotdxme += dx*dx+ dy*dy+ dz*dz;
            }
            n += 3;
        }
    }
    double dxdotdxall = 0;
	MPI_Allreduce(&dxdotdxme,&dxdotdxall,1,MPI_DOUBLE,MPI_SUM,world);
    dsaddle = dxdotdxall;
    
    //ADD PFs
    double sigma_gsq=0;
    sigma_gsq = width*width;
    double eng = 0;
    if(dxdotdxall < sigma_gsq){
        double delg = 1-dxdotdxall/sigma_gsq;
        double constant = 4*height/sigma_gsq;
        eng = delg*delg*height;
        if(domain->triclinic == 0){
            int n = 0;
            for (int i = 0; i < nlocal; i++){
                if (mask[i] & groupbit) {
                
                    //unwrap the coordinates to compare the real distance
                    xbox = (image[i] & IMGMASK) - IMGMAX;
                    ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
                    zbox = (image[i] >> IMG2BITS) - IMGMAX;
                    
                    dx = x[i][0] + xbox*xprd - xc[0] - x0[n];
                    dy = x[i][1] + ybox*yprd - xc[1] - x0[n+1];
                    dz = x[i][2] + zbox*zprd - xc[2] - x0[n+2];
            
                    if(dx <= 1e-12) dx = dx + random_num*pert;
                    if(dy <= 1e-12) dy = dy + random_num*pert;
                    if(dz <= 1e-12) dz = dz + random_num*pert;
            
                    f[i][0] += dx * delg * constant;
                    f[i][1] += dy * delg * constant;
                    f[i][2] += dz * delg * constant;
                }
                n += 3;
            }
        }else{
            int n = 0;
            for (int i = 0; i < nlocal; i++){
                if (mask[i] & groupbit) {
                    
                    xbox = (image[i] & IMGMASK) - IMGMAX;
                    ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
                    zbox = (image[i] >> IMG2BITS) - IMGMAX;
                    
                    dx = x[i][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox - xc[0] - x0[n];
                    dy = x[i][1] + h[1]*ybox + h[3]*zbox - xc[1] - x0[n+1];
                    dz = x[i][2] + h[2]*zbox - xc[2] - x0[n+2];
                    
                    if(dx <= 1e-12) dx = dx + random_num*pert;
                    if(dy <= 1e-12) dy = dy + random_num*pert;
                    if(dz <= 1e-12) dz = dz + random_num*pert;
                    
                    f[i][0] += dx * delg * constant;
                    f[i][1] += dy * delg * constant;
                    f[i][2] += dz * delg * constant;
                }
                n += 3;
            }
        }//else ends
    }
    return eng;
}

/*------------------------------Combine Penlaty Functions------------------------------------------- */
void FixABC::CombinePF(int itest)
{
    //if (comm->me == 0) fprintf(screen,"FixABC CombinePF nvector=%d store_no=%d itest=%d\n", nvector, store_no, itest);
    
    
    int n, i_min, ii, jj;
    double  dr_min, dx,dy,dz,xx,yy,zz,w1, w2, h1, h2, c1, dr0;
    double  dxmax, dymax, dzmax;

    int nlocal = atom->nlocal;
    double dxdotdxall = 0.0;
    double dxdotdxme = 0.0;
    
    dr_min = 1000000.0;
    i_min = itest;
    double *x0 = vectors[itest];
    
    //calculate the center of current pf to center of other pfs
    for ( ii = 0; ii < itest; ii++){
            
        dxdotdxme = dxdotdxall = 0.0;
        if ( pheight[ii] < 5*omega){
            double *y0 = vectors[ii];
            n = 0;
            for ( jj = 0; jj < nlocal; jj++){
                dx = y0[n] - x0[n];
                dy = y0[n+1] - x0[n+1];
                dz = y0[n+2] - x0[n+2];
                    
                dxdotdxme += dx*dx+dy*dy+dz*dz;
                n += 3;
            }
            MPI_Allreduce(&dxdotdxme,&dxdotdxall,1,MPI_DOUBLE,MPI_SUM,world);
            
            //Combine with the closest PF
            if(dr_min > dxdotdxall){
                dr_min  = dxdotdxall;
                i_min   = ii;
            }
        }
    }
        
    //  Combine 2PFs
    //  Combine the PFs iff the distance between two PFs are close enough (Check the paper)
    w1 = pwidth[itest];  h1 = pheight[itest];
    w2 = pwidth[i_min];  h2 = pheight[i_min];
    
    c1 = h1*w1/(h1*w1+h2*w2);
    dr0 = sqrt(dr_min);
    if( dr0 <= 0.5*(w1+w2)){
        if( w1< 5*wstart && w2 < 5*wstart && h1 < 5*omega && h2 < 5*omega){

            double *y0 = vectors[i_min];
            n = 0;
            for (jj = 0; jj < nlocal; jj++){
                xx = (1-c1)*y0[n] + c1*x0[n];
                yy = (1-c1)*y0[n+1] + c1*x0[n+1];
                zz = (1-c1)*y0[n+2] + c1*x0[n+2];
                
                y0[n] = xx;
                y0[n+1] = yy;
                y0[n+2] = zz;
                    
                x0[n] = 0.0;
                x0[n+1] = 0.0;
                x0[n+2] = 0.0;
                n += 3;
            }
                
            pwidth[itest] = 0.0;
            pheight[itest] = 0.0;
            pheight[i_min]  = (h1+h2)*(1-dr0/(w1 + w2));
            pwidth[i_min]   = MAX(w1,MAX(w2,(dr0+w2+w1)*0.5));
                
            nvector=nvector-1;
                
            if (comm->me == 0) fprintf(screen,"Combine %d and %d PF pheight[i_min]=%f pwidth[i_min]=%f \n",itest, i_min, pheight[i_min], pwidth[i_min]);
        }
    }
    
    
    
    
}

/*--------------------------------*/

int FixABC::store(int step)
{
	int nlocal = atom->nlocal;
	double **x = atom->x;
	int *mask = atom->mask;
	int *image = atom->image;
	double y[3];
	double xc[3];
	xxc(xc);
	
    if ( nvector >= store_no) add_vector();
    
    double *x0 = vectors[nvector];
	int n = 0;
	for (int i = 0; i < nlocal; i++){
			y[0] = y[1] = y[2] = 0.0;
			domain->unmap(x[i],image[i],y);
			x0[n] = y[0] - xc[0];
			x0[n+1] = y[1] - xc[1];
			x0[n+2] = y[2] - xc[2];
        
        n += 3;
	}
    
    pwidth[nvector] = wstart + 0.01*sigma*random_num;
    pheight[nvector] = omega;
    nvector++;
    
    if (comm->me == 0) fprintf(screen,"FixABC Store nvector=%d pwidth=%f pheight=%f \n",nvector, pwidth[nvector-1], pheight[nvector-1]);
    return 0;
}

/* ---------------------------------------------------------------------- */

void FixABC::xxc(double *c)
{
    double **x = atom->x;
    bigint natoms = atom->natoms;
    int nlocal = atom->nlocal;
    int *image = atom->image;
    int *type = atom->type;
    double *mass = atom->mass;
    
    double y[3];
    double cone[3];
    cone[0] = cone[1] = cone[2] = 0.0;
    
    
    double massone;
    double massproc, masstotal;
    
    massproc = 0.0;
    
    for (int i = 0; i < nlocal; i++){
        massone = mass[type[i]];
        y[0] = y[1] = y[2] = 0;
        domain->unmap(x[i],image[i],y);
        
        cone[0] += y[0] * massone;
        cone[1] += y[1] * massone;
        cone[2] += y[2] * massone;
        
        massproc = massproc + massone;
    }
    MPI_Allreduce(cone,c,3,MPI_DOUBLE,MPI_SUM,world);
    
    masstotal = 0.0;
    MPI_Allreduce(&massproc,&masstotal,1,MPI_DOUBLE,MPI_SUM,world);
    
    c[0] /= masstotal;
    c[1] /= masstotal;
    c[2] /= masstotal;
}
/* ---------------------------------------------------------------------- */







