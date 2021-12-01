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
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "input.h"
#include "output.h"
#include "thermo.h"
#include "error.h"
#include "math.h"
#include "random_park.h"
#include "write_data.h"

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "universe.h"
#include "domain.h"
#include "memory.h"
#include "group.h"
#include "neighbor.h"
#include "run.h"
#include "update.h"
#include "finish.h"
#include "timer.h"
#include "modify.h"
#include "variable.h"
#include "irregular.h"
#include "abc.h"
#include "fix.h"
#include "fix_abc.h"
#include "fix_sabc.h"
#include "min.h"
#include "minimize.h"
#include "compute.h"
#include "compute_pe.h"

#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"

using namespace LAMMPS_NS;
#define MAXLINE 256
#define CHUNK 1024
#define ATTRIBUTE_PERLINE 4
#define ROUND_2_INT(f) ((int)(f >= 0.0 ? (f + 0.5) : (f - 0.5)))

/* ---------------------------------------------------------------------- */

ABC::ABC(LAMMPS *lmp) : Pointers(lmp) {}
/*
 arg[0] = group
 arg[1] = omega
 arg[2] = sigma
 arg[3] = wstart
 arg[4] = nkeep
 arg[5] = temperature
 
 arg[6] = etol
 arg[7] = ftol
 arg[8] = maxiter
 arg[9] = maxeval
 
 arg[10] = maxpf
 arg[11] = ftolerance
 arg[12] = ebarrier
 arg[13] = dmin
 arg[14] = randpf
 arg[15] = Bcombine
 arg[16] = pert
 arg[17] = ndump
 arg[18] = Maxstrain
 arg[19] = maxstep
 #ABC group omega  sigma wstart  nlm     temp     (etol  ftol  maxiter maxeval)   maxpf  ftol    ebarrier  dmin  randpf Bcombine  pert     ndump   Maxstrain
 abc  center  3    500   10.0    2   ${temperature}   0    0.05   400     400      3000  5.0e-4      3.0     0.2   3333      1     1.0e-3	10		 1.0e-4

 */

/* ---------------------------------------------------------------------- */

void ABC::command(int narg, char **arg)
{
    nkeep = atoi(arg[4]);
    temperature = atof(arg[5]);
    
    maxpf = atoi(arg[10]);
    ftolerance = atof(arg[11]);
    ebarrier = atof(arg[12]);
    dmin = atof(arg[13]);
    
    seed_mc = atoi(arg[14]);
    random_mc = new RanPark(lmp,seed_mc);
    
    ndump = atoi(arg[17]);
    
    maxstrain = atof(arg[18]);
    
    igroup = group->find(arg[0]);
    if (igroup == -1) error->all(FLERR,"Could not find fix group ID");
    groupbit = group->bitmask[igroup];

    if(screen && comm->me == 0) fprintf(screen,"narg = %d, nkeep=%d temperature =%f  maxpf= %d ebarrier=%f dmin=%f ftolerance = %f ndump=%d maxstrain=%f domain->triclinic = %d h0=%f h1=%f h2=%f h3=%f h4=%f h5=%f domain->xy=%f groupbit=%d\n",narg, nkeep, temperature, maxpf, ebarrier,dmin, ftolerance, ndump, maxstrain, domain->triclinic, domain->h[0], domain->h[1],domain->h[2], domain->h[3], domain->h[4],domain->h[5],domain->xy,groupbit);
    
    update->restrict_output = 1;
    
    char **fixarg = new char*[3];
    fixarg[0] = (char *) "SABC";
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "SABC";
    modify->add_fix(3,fixarg);
    delete [] fixarg;
    fix_sabc = (FixSABC *) modify->fix[modify->nfix-1]; //setup FixSabc
    
    normalabc(narg,arg);
    update->restrict_output = 0;
}
/* ----------------------------------------------------------------------*/
void ABC::normalabc(int narg, char **arg)
{
    /********Initialization******/
    
    char **minarg;
    minarg = new char*[4];
    minarg[0] = arg[6];
    minarg[1] = arg[7];
    minarg[2] = arg[8];
    minarg[3] = arg[9];
    
    char **minarg1;
    minarg1 = new char*[4];
    minarg1[0] = (char *) "0";
    minarg1[1] = (char *) "0.05";
    minarg1[2] = (char *) "1000";
    minarg1[3] = (char *) "1000";    //minimize during pressure adjustment
    
    char **minarg2;
    minarg2 = new char*[4];
    minarg2[0] = (char *) "1.0e-9";
    minarg2[1] = (char *) "1.0e-9";
    minarg2[2] = (char *) "5000";;
    minarg2[3] = (char *) "5000";
    
    //Please note the format of the command!!!
    char **fixarg;
    if (narg == 19){
        fixarg = new char*[9];      //modify fixarg if need to add more arg
        fixarg[0] = (char *) "fixABC";
        fixarg[1] = arg[0];
        fixarg[2] = (char *) "abc";
        fixarg[3] = arg[1];
        fixarg[4] = arg[2];
        fixarg[5] = arg[3];
        fixarg[6] = arg[14];
        fixarg[7] = arg[15];
        fixarg[8] = arg[16];
        modify->add_fix(9,fixarg);
        delete [] fixarg;
    }else if (narg == 20){
        fixarg = new char*[10];
        fixarg[0] = (char *) "fixABC";
        fixarg[1] = arg[0];
        fixarg[2] = (char *) "abc";
        fixarg[3] = arg[1];
        fixarg[4] = arg[2];
        fixarg[5] = arg[3];
        fixarg[6] = arg[14];
        fixarg[7] = arg[15];
        fixarg[8] = arg[16];
        fixarg[9] = arg[19];
        modify->add_fix(10,fixarg);
        delete [] fixarg;
    }
    fix_abc = (FixABC *) modify->fix[modify->nfix-1]; //Initialize Fix_abc
    
    
    
    /****------Set up LMP.abc-----***/
    fp = NULL;
    if (comm->me ==0){
        fp = fopen("LMP.abc","a");
        if(fp==NULL){
            fp = fopen("LMP.abc","w");
        }
    }
    
    double *energy_steps, *enmax_steps, *press_steps;
    double  pressx_steps, pressy_steps, pressz_steps, zhi,zlo;
    double dr_minpair, dr_maxpair, MSD_abc, current_energy, enmax, temp, psaddlemax, pminimummin, einitial, deng, DE, diff_E;
    int nlocal, ipf;
    
    //for kMC
    int   flag_peak, id, temp1;
    int natoms = atom->natoms;
    
    
    //MAIN FUNCTION. Deform the configuration by 10^-4/step.
    for(int shear_step = 0; shear_step < 10000; shear_step++){
    
        /****Setup for ABC searching***/
    
        psaddlemax = -1e20;
        pminimummin = 1e20;
    
        dr_minpair = 100000.0;
        dr_maxpair = 0.0;
        lm_no = 0;
        store_lm = 0;
        DE = 0.0;
        diff_E = 0.0;
    
        energy_steps = NULL;
        memory->grow(energy_steps, maxpf, "abc:energy_steps");
        
        enmax_steps = NULL;
        memory->grow(enmax_steps,  maxpf,"abc:energy_steps");
    
        press_steps = NULL;
        memory->grow(press_steps,  maxpf,"abc:energy_steps");
    
    
    
        /*---------------------DEFORM------------------*/
        if(screen && comm->me == 0) fprintf(screen,"-----------ABC DEFORM-----------\n");
        
        double strain;
        strain = maxstrain;
        
        dilation[0] = 1.0;
        dilation[1] = 1.0;
        dilation[2] = 1.0+strain;
        
        remap();
        
        //minimize the system
        fix_abc->set_Baddpf(0);
        min(4,minarg1);
    
        /*----------Initial energy basin------------*/
        //record the initial configuration
        char *str = new char[150];
        
        if(shear_step % ndump == 0){
            
            if (comm->me == 0) fprintf(screen,"domain->box_too_small_check()\n");
            sprintf(str,"write_data    ./ener/data_%d.init",shear_step);
            input->one(str);
        }
        
        
        //output the coordinates of the lm for neb calculation
        //write(lm_no);
    
        //store initial state (3D search)
        storeabc();
    
        //Thermal informations
        id = modify->find_compute((char *)"10");
        if (id < 0)   error->all(FLERR,"abc_pe ID for abc_pe does not exist");
        current_energy = modify->compute[id]->scalar;
        einitial = current_energy;
    	pminimummin = current_energy;
    
        id = modify->find_compute((char *)"11");
        if (id < 0)   error->all(FLERR,"compute ID 11 for ABC does not exist");
        pressx_steps = modify->compute[id]->scalar;
    
    
        id = modify->find_compute((char *)"12");
        if (id < 0)   error->all(FLERR,"compute ID 12 for ABC does not exist");
        pressy_steps = modify->compute[id]->scalar;
    
        id = modify->find_compute((char *)"13");
        if (id < 0)   error->all(FLERR,"compute ID 13 for ABC does not exist");
        pressz_steps = modify->compute[id]->scalar;
    

        id = input->variable->find((char *)"p8");
        if (id < 0)   error->all(FLERR,"Could not find variable name p8");
        zlo= input->variable->compute_equal(id);
    
        id = input->variable->find((char *)"p9");
        if (id < 0)   error->all(FLERR,"Could not find variable name p8");
        zhi = input->variable->compute_equal(id);
    
        energy_steps[lm_no-1] = current_energy;
        enmax_steps[lm_no-1] = 0;
        press_steps[lm_no-1] = pressz_steps;
    
        if (comm->me ==0) {
            if(fp) fprintf(fp,"50000  50000  50000  %15.8g  %15.8g  %15.8g  %15.8g                0        0         %15.8g  %15.8g                 0  %15.8g\n ", energy_steps[lm_no-1], pressx_steps, pressy_steps, pressz_steps, zlo, zhi, strain);
            
            //if(screen) fprintf(screen,"50000  50000  50000  %15.8g  %15.8g  %15.8g  %15.8g      0        0       %15.8g  %15.8g                 0  %15.8g\n ", energy_steps[lm_no-1], pressx_steps, pressy_steps, pressz_steps, zlo, zhi, strain);
        }
    
        /*****------ABC SEARCH LOOP-------**/
        for (int igaus = 0; igaus < maxpf; igaus++){
        
            //For Debug
            //Calculate f_org (without PFs before minimization)
            nlocal = atom->nlocal;
            double **f = atom->f;
            double f_all = 0.0;
            for(int i = 0; i < nlocal; i++){
                for(int s = 0; s < 3; s++){
                    f_all = f_all + f[i][s]*f[i][s];
                }
            }
            
            double f_org = 0.0;
            MPI_Allreduce(&f_all,&f_org,1,MPI_DOUBLE,MPI_SUM,world);
            f_org = f_org/natoms;
            
            
            
            //add pf and minimize the system
            fix_abc->set_Baddpf(1);
            min(4,minarg);
            
            //Calculate f_aft (without PFs after minimization)
            nlocal = atom->nlocal;
            f = atom->f;
            f_all = 0.0;
            for(int i = 0; i < nlocal; i++){
                for(int s = 0; s < 3; s++){
                    f_all = f_all + f[i][s]*f[i][s];
                }
            }
            double f_after = 0.0;
            MPI_Allreduce(&f_all,&f_after,1,MPI_DOUBLE,MPI_SUM,world);
            f_after = f_after/natoms;
            if (comm->me == 0) fprintf(screen,"igaus=%d f_org %15.8g f_after %15.8g!\n",igaus, f_org, f_after);
            
            
            
            /*************If it is a local minimum******************/
            if(f_after <= ftolerance){
            //save local minima
                dr_minpair = 1000000.0;
                dr_maxpair = 0.0;
                
                if(store_lm >= 1){ //calculate the distance from previous local minima
                
                    double *h = domain->h;
                    double xprd = domain->xprd;
                    double yprd = domain->yprd;
                    double zprd = domain->zprd;
                    int xbox,ybox,zbox;

                    double **x = atom->x;
                    imageint *image = atom->image;
                    double xc[3];
                    xxc(xc);
                    
                    double *cm0 = fix_sabc->vstore;
                    
                    int ii = 0;
                    while (ii < store_lm){
                        
                        //if (comm->me == 0) fprintf(screen,"lm_no = %d ilm_no = %d!\n",lm_no, ii);
                        
                        double *x0 = fix_sabc->request_vector(ii);
                        double dx, dy, dz,dxdotdxme, dxdotdxall;
                        dxdotdxme = 0.0;
                        
                        //calculate the distance between current lm and all other lms
                        if(domain->triclinic == 0){
                            int n = 0;
                            for (int i = 0; i < nlocal; i++){
                                    xbox = (image[i] & IMGMASK) - IMGMAX;
                                    ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
                                    zbox = (image[i] >> IMG2BITS) - IMGMAX;
                                
                                    dx = x[i][0] + xbox*xprd - xc[0] - x0[n] + cm0[3*ii+0];
                                    dy = x[i][1] + ybox*yprd - xc[1] - x0[n+1] + cm0[3*ii+1];
                                    dz = x[i][2] + zbox*zprd - xc[2] - x0[n+2] + cm0[3*ii+2];
                                
                                    dxdotdxme += dx*dx+ dy*dy+ dz*dz;
                                    n += 3;
                            }
                        }else{
                            int n = 0;
                            for (int i = 0; i < nlocal; i++){
                                    xbox = (image[i] & IMGMASK) - IMGMAX;
                                    ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
                                    zbox = (image[i] >> IMG2BITS) - IMGMAX;
                                
                                    dx = x[i][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox - xc[0] - x0[n]+ cm0[3*ii+0];
                                    dy = x[i][1] + h[1]*ybox + h[3]*zbox - xc[1] - x0[n+1]+ cm0[3*ii+1];
                                    dz = x[i][2] + h[2]*zbox - xc[2] - x0[n+2]+ cm0[3*ii+2];
                                
                                    //if (comm->me == 0) fprintf(screen,"dx %f %f %f \n",dx, dy, dz);
                                
                                    dxdotdxme += dx*dx+ dy*dy+ dz*dz;
                                    n += 3;
                            }
                        }

                        dxdotdxall = 0.0;
                        MPI_Allreduce(&dxdotdxme,&dxdotdxall,1,MPI_DOUBLE,MPI_SUM,world);
                        dxdotdxall = dxdotdxall/natoms;
                        
                        if(dxdotdxall < dr_minpair) {
                            dr_minpair = dxdotdxall;
                        }
                        
                        //check whether the code the correct
                        if(dxdotdxall > dr_maxpair) {
                            dr_maxpair = dxdotdxall;
                        }
                        
                        
                        //FOR DEBUG
                        // if (comm->me == 0) fprintf(screen,"lm_no = %d ilm_no = %d dxdotdxall = %20.15g msd = %20.15g!\n",lm_no, ii, dxdotdxall, MSD_abc);
                        
                        if( ii == store_lm - 1){
                            MSD_abc = dxdotdxall;
                            //if (comm->me == 0) fprintf(screen,"lm_no = %d ilm_no = %d dxdotdxall = %20.15g msd = %20.15g!\n",lm_no, ii, dxdotdxall, MSD_abc);
                        }
                        
                        ii = ii + 1;
                        
                    }//for ii ends
                
                } //if lm_no >= 1 ends
                
                
                //THERMO INFORMATIONS
                id = modify->find_compute((char *)"10");
                if (id < 0)   error->all(FLERR,"abc_pe ID for abc_pe does not exist");
                current_energy = modify->compute[id]->scalar;
                
                id = modify->find_compute((char *)"11");
                if (id < 0)   error->all(FLERR,"compute ID 11 for ABC does not exist");
                pressx_steps = modify->compute[id]->scalar;
                
                
                id = modify->find_compute((char *)"12");
                if (id < 0)   error->all(FLERR,"compute ID 12 for ABC does not exist");
                pressy_steps = modify->compute[id]->scalar;
                
                id = modify->find_compute((char *)"13");
                if (id < 0)   error->all(FLERR,"compute ID 13 for ABC does not exist");
                pressz_steps = modify->compute[id]->scalar;
                
                
                id = input->variable->find((char *)"p8");
                if (id < 0)   error->all(FLERR,"Could not find variable name p8");
                zlo= input->variable->compute_equal(id);
                
                id = input->variable->find((char *)"p9");
                if (id < 0)   error->all(FLERR,"Could not find variable name p8");
                zhi = input->variable->compute_equal(id);
                
                
            
                enmax = fix_abc->enmax;
                double etemp = enmax - einitial;
                if(etemp > DE) DE = etemp;
                deng = current_energy - einitial;
               
                if (comm->me == 0) fprintf(screen,"store_lm =%d MSD_abc=%f DE=%f deng=%f dr_minpair=%f dmin=%f !\n", store_lm, MSD_abc, DE, deng, dr_minpair, dmin);
                
                //IF it is a new local minimum
                if(dr_minpair > dmin && DE - deng > 0.1){
                    
                    //add pf and minimize the system
                    fix_abc->set_Baddpf(0);
                    min(4,minarg);
                    
                    
                    //THERMO INFORMATIONS
                    id = modify->find_compute((char *)"10");
                    if (id < 0)   error->all(FLERR,"abc_pe ID for abc_pe does not exist");
                    current_energy = modify->compute[id]->scalar;
                    
                    id = modify->find_compute((char *)"11");
                    if (id < 0)   error->all(FLERR,"compute ID 11 for ABC does not exist");
                    pressx_steps = modify->compute[id]->scalar;
                    
                    
                    id = modify->find_compute((char *)"12");
                    if (id < 0)   error->all(FLERR,"compute ID 12 for ABC does not exist");
                    pressy_steps = modify->compute[id]->scalar;
                    
                    id = modify->find_compute((char *)"13");
                    if (id < 0)   error->all(FLERR,"compute ID 13 for ABC does not exist");
                    pressz_steps = modify->compute[id]->scalar;
                    
                    
                    enmax = fix_abc->enmax;
                    double etemp = enmax - einitial;
                    if(etemp > DE) DE = etemp;
                    deng = current_energy - einitial;
                    
                    
                    //output the coordinates of the lm for neb calculation
                    //write(lm_no);
                    
                    //store the information of lm for new lm test!
                    storeabc();
                    
                    energy_steps[lm_no-1] = current_energy;
                    enmax_steps[lm_no-1] = DE;
                    press_steps[lm_no-1] = pressz_steps;
                    
                    //compute energy barrier
                    if(enmax > psaddlemax){             //store highest sadle point
                        psaddlemax = enmax;
                    }

					diff_E = psaddlemax - pminimummin;

            		//if(comm->me ==0 && screen) fprintf(screen,"%5d  %5d  %5d  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g %15.8g  %15.8g %15.8g %15.8g \n ", igaus, lm_no, store_lm,  energy_steps[lm_no-1], pressx_steps, pressy_steps, pressz_steps, deng, DE, dr_minpair, zlo, zhi, diff_E);
                    
                    fflush(fp);
                    if(comm->me ==0 && fp) fprintf(fp,"%5d  %5d  %5d  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g %15.8g %15.8g %15.8g\n ", igaus, lm_no, store_lm, energy_steps[lm_no-1], pressx_steps, pressy_steps, pressz_steps, deng, DE, dr_minpair,zlo, zhi, diff_E);
                    
                    DE = 0.0;
                    deng = 0.0;
                    einitial = current_energy;
					
					if(current_energy < pminimummin){
						pminimummin = current_energy;
					}
                    
                }else if( igaus%5 == 0 ){
                    
                    //FOR DEBUG
                    //if(comm->me ==0 && screen) fprintf(screen,"%5d  %5d  %5d  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g %15.8g %15.8g %15.8g\n ", igaus, lm_no, store_lm, current_energy, pressx_steps, pressy_steps, pressz_steps, deng, DE, dr_minpair,zlo, zhi, diff_E);
                    
                     fflush(fp);
                     if(comm->me ==0 && fp) fprintf(fp,"%5d  %5d  %5d  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g  %15.8g %15.8g %15.8g %15.8g\n ", igaus, lm_no, store_lm, current_energy, pressx_steps, pressy_steps, pressz_steps, deng, DE, dr_minpair,zlo, zhi, diff_E);
                }
                
                    
            }//f_all <= ftolerance ends
            

            //Stop ABC!
            if(diff_E > ebarrier || lm_no == nkeep) {
            //if(lm_no == nkeep) {
                if (comm->me == 0) fprintf(screen,"BREAK THE LOOP!\n");
                break;
            }
        } //for igaus end
        
        
        /******KMC to compute final state*******/
        if(screen && comm->me == 0) fprintf(screen,"Start KMC \n");
         
         if(lm_no <= 2){
             flag_peak = 1;
         }else {
             //choose the one with lowest stress value (You can change this criteria for your simulations)
             double lowest_energy = 1e10;
             for(int i = 1; i < lm_no; i++){
                 if(press_steps[i] < lowest_energy){
                     lowest_energy = press_steps[i];
                     flag_peak = i;
                 }
             }
         }
        
        if(comm->me == 0){
            fprintf(screen, "pressz %f %f flag_peak = %d\n", press_steps[flag_peak], energy_steps[flag_peak],flag_peak);
        }
        
        //clear the stored PF informations
        fix_abc->clear_vector();
        
        backtrack(flag_peak);
        
        
        //clear all the vectors and start a new step
        memory->destroy(energy_steps);
        memory->destroy(enmax_steps);
        memory->destroy(press_steps);
    
    } //end for shear_step
    
	
	modify->delete_fix((char *) "fixABC");
    modify->delete_fix((char *) "SABC");
    
    if (fp && comm->me == 0) fclose(fp);
    delete [] minarg;
    delete [] minarg1;
    delete [] minarg2;
}

/* ---------------------------------------------------------------------- */

void ABC::backtrack(int flag_peak)
{
    if (comm->me == 0) fprintf(screen,"ABC Backtrack flag_peak=%d\n", flag_peak);
    
    double *x0 = fix_sabc->request_vector(flag_peak);
    double **x = atom->x;
    int nlocal = atom->nlocal;
    int *image = atom->image;
    
    int n = 0;
    for (int i = 0; i < nlocal; i++){
        x[i][0] = x0[n];
        x[i][1] = x0[n+1];
        x[i][2] = x0[n+2];
        n += 3;
        image[i] = ((imageint) IMGMAX << IMG2BITS) |((imageint) IMGMAX << IMGBITS) | IMGMAX;
        domain->remap(x[i],image[i]);
    }
    
    //clear all the vectors and start a new step
    fix_sabc->clear_vector();
    
    int triclinic = domain->triclinic;
    if (modify->n_min_pre_exchange) modify->min_pre_exchange();
    if(triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
    }
    
    if (comm->me == 0) fprintf(screen,"com exchanges\n");
    
    comm->exchange();
    
    if (comm->me == 0) fprintf(screen,"atom sort\n");
    
    if (atom->sortfreq > 0 && update->ntimestep >= atom->nextsort) atom->sort();
    
    if (comm->me == 0) fprintf(screen,"comm borders\n");
    
    comm->borders();
    
    if(triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    
    if (comm->me == 0) fprintf(screen,"neighbor build\n");
    neighbor->build();
    
    
    // check if any atoms were lost
    bigint ncount;
    bigint nblocal = atom->nlocal;
    MPI_Allreduce(&nblocal,&ncount,1,MPI_LMP_BIGINT,MPI_SUM,world);
    
    if (comm->me == 0) fprintf(screen,"NCOUNT=%d\n",ncount);
    
    if (ncount != atom->natoms && comm->me == 0) {
        char str[128];
        sprintf(str,"Lost atoms via displace_atoms: original " BIGINT_FORMAT
                " current " BIGINT_FORMAT,atom->natoms,ncount);
        error->warning(FLERR,str);
    }
    
    
    if (comm->me == 0) fprintf(screen,"ABC Backtrack ends\n");
}



/* ----------------------------------------------------------------------*/
void ABC::write(int lm_no){
    
    if (comm->me == 0 && screen)
        fprintf(screen,"System init for write_data ...\n");
    
    // move atoms to new processors before writing file
    // do setup_pre_exchange to force update of per-atom info if needed
    // enforce PBC in case atoms are outside box
    // call borders() to rebuild atom map since exchange() destroys map
    lmp->init();
    modify->setup_pre_exchange();
    
    //if (comm->me == 0) fprintf(screen,"domain->x2lamda(atom->nlocal)\n");
    
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    
    //if (comm->me == 0) fprintf(screen," domain->pbc()\n");
    domain->pbc();
    
    //if (comm->me == 0) fprintf(screen,"domain->reset_box()\n");
    domain->reset_box();
    
    //if (comm->me == 0) fprintf(screen,"comm->setup()\n");
    comm->setup();
    
    //if (comm->me == 0) fprintf(screen,"comm->exchange()\n");
    comm->exchange();
    
    //if (comm->me == 0) fprintf(screen,"comm->borders()\n");
    comm->borders();
    
    //if (comm->me == 0) fprintf(screen,"domain->lamda2x(atom->nlocal+atom->nghost)\n");
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    
    //if (comm->me == 0) fprintf(screen,"domain->image_check()\n");
    domain->image_check();
    
    //if (comm->me == 0) fprintf(screen,"domain->box_too_small_check()\n");
    domain->box_too_small_check();
    
    
    char *file = new char[30];
    sprintf(file,"./ener/abc""%d"".data",lm_no);
    
    bigint nblocal = atom->nlocal;
    bigint natoms;
    MPI_Allreduce(&nblocal,&natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
    if (natoms != atom->natoms)
        error->all(FLERR,"Atom count is inconsistent, cannot write data file");
    
    
    //OPEN FILE TO WRITE
    FILE *fp2 = NULL;
    if (comm->me == 0) {
        fp2 = fopen(file,"w");
        if (fp2 == NULL) {
            char str[128];
            sprintf(str,"Cannot open data file %s",file);
            error->one(FLERR,str);
        }
    }
    
    //write atoms coordinates to file
    if(natoms){
        
        int ncol = atom->avec->size_data_atom + 1;
        int sendrow = atom->nlocal;
        int maxrow;
        MPI_Allreduce(&sendrow,&maxrow,1,MPI_INT,MPI_MAX,world);
        
        double **buf;
        if (comm->me == 0) memory->create(buf,MAX(1,maxrow),ncol,"write_data:buf");
        else memory->create(buf,MAX(1,sendrow),ncol,"write_data:buf");
        
        tagint *tag = atom->tag;
        int nlocal = atom->nlocal;
        double *h = domain->h;
        double xprd = domain->xprd;
        double yprd = domain->yprd;
        double zprd = domain->zprd;
        int xbox,ybox,zbox;
        double **x = atom->x;
        imageint *image = atom->image;
        if(domain->triclinic == 0){
            for (int i = 0; i < nlocal; i++){
                
                xbox = (image[i] & IMGMASK) - IMGMAX;
                ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
                zbox = (image[i] >> IMG2BITS) - IMGMAX;
                
                buf[i][0] = x[i][0] + xbox*xprd;
                buf[i][1] = x[i][1] + ybox*yprd;
                buf[i][2] = x[i][2] + zbox*zprd;
                buf[i][3] = double(tag[i]);
            }
        }else{
            for (int i = 0; i < nlocal; i++){
                
                xbox = (image[i] & IMGMASK) - IMGMAX;
                ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
                zbox = (image[i] >> IMG2BITS) - IMGMAX;
                
                buf[i][0] = x[i][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox;
                buf[i][1] = x[i][1] + h[1]*ybox + h[3]*zbox;
                buf[i][2] = x[i][2] + h[2]*zbox;
                buf[i][3] = double(tag[i]);
            }
        }
        
        // write one chunk of atoms per proc to file
        // proc 0 pings each proc, receives its chunk, writes to file
        // all other procs wait for ping, send their chunk to proc 0
        int tmp,recvrow;
        MPI_Status status;
        MPI_Request request;
        
        if (comm->me == 0) {
            
            fprintf(fp2,"%d\n",atom->natoms);
            //write box dimensions to ener%d.data
            //fprintf(fp2,"%-1.16e %-1.16e xlo xhi\n",domain->boxlo[0],domain->boxhi[0]);
            //fprintf(fp2,"%-1.16e %-1.16e ylo yhi\n",domain->boxlo[1],domain->boxhi[1]);
            //fprintf(fp2,"%-1.16e %-1.16e zlo zhi\n",domain->boxlo[2],domain->boxhi[2]);
            
            //for shear
            //fprintf(fp2,"%-1.16e %-1.16e %-1.16e xy xz yz\n",domain->xy,domain->xz,domain->yz);
            
            for (int iproc = 0; iproc < comm->nprocs; iproc++) {
                if (iproc) {
                    MPI_Irecv(&buf[0][0],maxrow*ncol,MPI_DOUBLE,iproc,0,world,&request);
                    
                    MPI_Send(&tmp,0,MPI_INT,iproc,0,world);
                    MPI_Wait(&request,&status);
                    MPI_Get_count(&status,MPI_DOUBLE,&recvrow);
                    recvrow /= ncol;
                } else recvrow = sendrow;
                
                for (int i = 0; i < recvrow; i++)
                    fprintf(fp2,"%5d  %15.8g  %15.8g  %15.8g\n",tagint(buf[i][3]), buf[i][0],buf[i][1], buf[i][2]);
            }
        } else {
            MPI_Recv(&tmp,0,MPI_INT,0,0,world,&status);
            MPI_Rsend(&buf[0][0],sendrow*ncol,MPI_DOUBLE,0,0,world);
        }
        // open data file
        if (comm->me == 0 && fp2) {
            fprintf(fp2,"\n\n\n");
            fprintf(fp2,"\nVelocities\n\n");
            fclose(fp2);
        }
        
        
        memory->destroy(buf);
    }
    
    
}

/* ----------------------------------------------------------------------*/
void ABC::min(int narg, char **arg)
{
    
    neighbor->every = 1;
    neighbor->delay = 0;
    neighbor->dist_check = 1;
    
    if (comm->me == 0) fprintf(screen," \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    if (comm->me == 0) fprintf(screen,"ABC Call_min %s %s %s %s\n", arg[0], arg[1], arg[2], arg[3]);
    
    if (narg != 4) error->all(FLERR,"Illegal minimize command");
    
    if (domain->box_exist == 0)
        error->all(FLERR,"Minimize command before simulation box is defined");
    
    update->etol = atof(arg[0]);
    update->ftol = atof(arg[1]);
    update->nsteps = atoi(arg[2]);
    update->max_eval = atoi(arg[3]);
    
    if (update->etol < 0.0 || update->ftol < 0.0)
        error->all(FLERR,"Illegal minimize command");
    
    update->whichflag = 2;
    update->beginstep = update->firststep = update->ntimestep;
    update->endstep = update->laststep = update->firststep + update->nsteps;
    if (update->laststep < 0 || update->laststep > MAXBIGINT)
        error->all(FLERR,"Too many iterations");
    
    lmp->init();
    update->minimize->setup();
    timer->init();
    timer->barrier_start();
    update->minimize->run(update->nsteps);
    timer->barrier_stop();
    update->minimize->cleanup();
    
    Finish finish(lmp);
    //finish.end(1);
    
    update->whichflag = 0;
    update->firststep = update->laststep = 0;
    update->beginstep = update->endstep = 0;
    
    //if (comm->me == 0) fprintf(screen,"ABC Call_min Finished\n");
}

/* ----------------------------------------------------------------------*/
void ABC::xxc(double *c)
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

/* ----------------------------------------------------------------------*/
void ABC::storeabc()
{
    
    if (comm->me == 0) fprintf(screen,"STORE ABC lm_no = %d!\n", lm_no);
    fix_sabc->add_vector();
    
    double *x0 =  fix_sabc->request_vector(store_lm);
    double *cm0 =  fix_sabc->vstore;
    
    double **x = atom->x;
    int *mask = atom->mask;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;
    double cm[3], y[3];
    xxc(cm);
    
    cm0[3*store_lm+0] = cm[0];
    cm0[3*store_lm+1] = cm[1];
    cm0[3*store_lm+2] = cm[2];
    
    int n = 0;
    for (int i = 0; i < nlocal; i++){
        
        y[0] = y[1] = y[2] = 0.0;
        domain->unmap(x[i],image[i],y);
        
        x0[n+0] = y[0] ;
        x0[n+1] = y[1] ;
        x0[n+2] = y[2] ;
        n = n + 3;
    }
    store_lm++;
    lm_no++;
}

/* ----------------------------------------------------------------------*/
void ABC::remap()
{
    
    
    int i;
    double oldlo,oldhi, newlo, newhi, ctr, difflo, diffhi, tilt_target, xprd, current;
    
    int nlocal = atom->nlocal;
    double **x = atom->x;
    
	//REMAP_X convert atoms and rigid bodies to lamda coords
    
    for (i = 0; i < nlocal; i++)
        domain->x2lamda(x[i],x[i]);
    
    //FOR SHEAR
    /*xprd = domain->boxhi[1] - domain->boxlo[1];
    current = domain->xy/xprd;
    tilt_target = domain->xy + delt*xprd;
    
    while (tilt_target/xprd - current > 0.0)
        tilt_target -= xprd;
    while (tilt_target/xprd - current < 0.0)
        tilt_target += xprd;
    if (fabs(tilt_target/xprd - 1.0 - current) < fabs(tilt_target/xprd - current))
        tilt_target -= xprd;
    
    
    domain->xy = tilt_target;
    if (comm->me == 0) fprintf(screen,"domain->xy = %f!\n", domain->xy);*/

	 
    //FOR TENSION
    for (i = 0; i < 3; i++) {
        oldlo = domain->boxlo[i];
        oldhi = domain->boxhi[i];
        
        ctr = (oldhi + oldlo)/2;
        
        difflo = - (oldhi - oldlo)*dilation[i]/2;   //negative
        diffhi = - difflo;                          //positive
        newlo = difflo;
        newhi = diffhi;
        
        domain->boxlo[i] = newlo;
        domain->boxhi[i] = newhi;
    }
    
    domain->set_global_box();
    domain->set_local_box();
    
	
	// convert atoms and rigid bodies back to box coords
    x = atom->x;
    nlocal = atom->nlocal;
    for (i = 0; i < nlocal; i++)    domain->lamda2x(x[i],x[i]);
    
    if (force->kspace) force->kspace->setup();
    
    /*int triclinic = domain->triclinic;
    if (modify->n_min_pre_exchange) modify->min_pre_exchange();
    
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
    }
    comm->exchange();
    if (atom->sortfreq > 0 &&
        update->ntimestep >= atom->nextsort) atom->sort();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    neighbor->build();*/
    
    if(screen && comm->me == 0) fprintf(screen,"-----------ABC DEFORM ENDS-----------\n");
    
}

/* --------------------------THERMAL OUTPUT------------------------------*/
double ABC::compute_evdwl()
{
    //if(screen && comm->me == 0) fprintf(screen,"ABC compute_evdwl!!\n");
    double tmp = 0.0;
    double dvalue = 0.0;
    if (force->pair) tmp += force->pair->eng_vdwl;
    MPI_Allreduce(&tmp,&dvalue,1,MPI_DOUBLE,MPI_SUM,world);
    
    if (force->pair && force->pair->tail_flag) {
        double volume = domain->xprd * domain->yprd * domain->zprd;
        dvalue += force->pair->etail / volume;
    }
    return dvalue;
}

/* ----------------------------------------------------------------------*/
double ABC::compute_ecoul()
{
    //if(screen && comm->me == 0) fprintf(screen,"ABC compute_ecoul!!\n");
    double tmp = 0.0;
    double dvalue = 0.0;
    if (force->pair) tmp += force->pair->eng_coul;
    MPI_Allreduce(&tmp,&dvalue,1,MPI_DOUBLE,MPI_SUM,world);
    
    return dvalue;
}

/* ----------------------------------------------------------------------*/
double ABC::compute_ebond()
{
    //if(screen && comm->me == 0) fprintf(screen,"ABC compute_ebond!!\n");
    double dvalue = 0.0;
    
    if (force->bond) {
        double tmp = force->bond->energy;
        MPI_Allreduce(&tmp,&dvalue,1,MPI_DOUBLE,MPI_SUM,world);
    } else dvalue = 0.0;
    
    return dvalue;
}

/* ----------------------------------------------------------------------*/
double ABC::compute_eangle()
{
    //if(screen && comm->me == 0) fprintf(screen,"ABC compute_eangle!!\n");
    double dvalue = 0.0;
    
    if (force->angle) {
        double tmp = force->angle->energy;
        MPI_Allreduce(&tmp,&dvalue,1,MPI_DOUBLE,MPI_SUM,world);
    } else dvalue = 0.0;
    
    return dvalue;
}

/* ----------------------------------------------------------------------*/
double ABC::compute_edihed()
{
    //if(screen && comm->me == 0) fprintf(screen,"ABC compute_edihed!!\n");
    double dvalue = 0.0;
    
    if (force->dihedral) {
        double tmp = force->dihedral->energy;
        MPI_Allreduce(&tmp,&dvalue,1,MPI_DOUBLE,MPI_SUM,world);
    } else dvalue = 0.0;
    
    return dvalue;
}

/* ----------------------------------------------------------------------*/
double ABC::compute_eimp()
{
    //if(screen && comm->me == 0) fprintf(screen,"ABC compute_eimp!!\n");
    double dvalue = 0.0;
    
    if (force->improper) {
        double tmp = force->improper->energy;
        MPI_Allreduce(&tmp,&dvalue,1,MPI_DOUBLE,MPI_SUM,world);
    } else dvalue = 0.0;
    
    return dvalue;
}

/* ----------------------------------------------------------------------*/
double ABC::compute_elong()
{
    //if(screen && comm->me == 0) fprintf(screen,"ABC compute_elong!!\n");
    double dvalue = 0.0;
    
    if (force->kspace) {
        dvalue = force->kspace->energy;
    } else dvalue = 0.0;
    
    return dvalue;
}

/* ----------------------------------------------------------------------*/
double ABC::compute_density()
{
    //if(screen && comm->me == 0) fprintf(screen,"ABC compute_density!!\n");
    double mass = group->mass(0);
    double dvalue = 0.0;
    double volume = 0.0;
    
    if (domain->dimension == 3)
        volume = domain->xprd * domain->yprd * domain->zprd;
    else
        volume = domain->xprd * domain->yprd;
    
    dvalue = force->mv2d * mass/volume;
    return dvalue;
}










