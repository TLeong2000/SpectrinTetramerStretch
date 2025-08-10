/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_arc_length_restraint.h"

#include "math.h"
#include "stdlib.h"
#include "atom.h"
#include "comm.h"
#include "molecule.h"
#include "error.h"
#include "modify.h"
#include "string.h"
#include "force.h"
#include "domain.h"
#include "memory.h"
#include "neighbor.h"

#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixArcLengthRestraint::FixArcLengthRestraint(LAMMPS *lmp, int narg, char **arg) : 
   Fix(lmp, narg, arg) 
{
   global_freq = 1;
   scalar_flag = 1;
   energy_global_flag = 1;
   extscalar = 1;
   thermo_energy = 1;

   debug_mode = false;

   if (narg < 6) error->all(FLERR, "Insufficient args for fix arclengthrestraint command.");

   // arg[3] is supposed to be the name of the molecule template that the user
   // wants to restrain; see the "ID" section of the molecule command
   // (which has nothing directly to do with molecule indexing)
   // atom->find_molecule(arg[3]) returns the molecule template ID associated
   // with the molecule template name stored in arg[3]
   imol = atom->find_molecule(arg[3]);

   // k is the restraining force proportionality constant
   // Because all inputs are char arrays, we use utils::numeric
   // to convert the char array to a double.
   k = utils::numeric(FLERR,arg[4],false,lmp);

   printf("Restraining force constant: %f\n", k);

   // equiLength is the equilibrium length that we want to restrain
   // the arc length of the chain to.
   equiLength = utils::numeric(FLERR,arg[5],false,lmp);

   printf("equiLength: %f\n\n", equiLength);

   // Get number of atoms for single copy of the molecule
   napmol = (atom->molecules[imol])->natoms;

   // Because the molecule is linear, the number of bonds for a single copy is
   // equal to the number of atoms minus 1
   nbpmol = napmol - 1;

   // Get total number of atoms in system
   natoms = atom->natoms;

   // Number of molecules in system is equal to number of atoms in system
   // divided by number of atoms for a single copy of the molecule
   nmols = natoms / napmol;
}

/* ---------------------------------------------------------------------- */

void FixArcLengthRestraint::init()
{
   neighbor->add_request(this);
}

/* ---------------------------------------------------------------------- */

int FixArcLengthRestraint::setmask()
{
   int mask = 0;
   mask |= FixConst::POST_FORCE;
   mask |= FixConst::MIN_POST_FORCE;
   return mask;
}

/* ---------------------------------------------------------------------- */

void FixArcLengthRestraint::post_force(int /*vflag*/)
{ 
   int rank;
   MPI_Comm_rank(world, &rank);

   if (debug_mode) {
      if (rank == 0) {
         printf("\n### Begin calculation of restraint forces called by post_force() ###\n");
      }
   }   

   set_forces();

   if (debug_mode) {
      if (rank == 0) {
         printf("\n### Restraining forces called by post_force() is now complete. ###\n");
      }
   }

}

/* ---------------------------------------------------------------------- */

void FixArcLengthRestraint::min_post_force(int vflag) {
   int rank;
   MPI_Comm_rank(world, &rank);

   if (debug_mode) {
      if (rank == 0) {
         printf("\n### Begin calculations of forces called by min_post_force() ###\n");
      }
   }

   set_forces();

   if (debug_mode) {
      if (rank == 0) {
	  printf("\n### Restraining forces called by min_post_force complete. ###\n");
      }
   }

}

/* ------------------------------------------------------------------------- */

double FixArcLengthRestraint::compute_scalar() {
   int rank;
   MPI_Comm_rank(world, &rank);

   if (debug_mode) {
      if (rank == 0) {
         printf("\n### Calculation of global restraint energy called with compute_scalar() ###\n");
         // printf("The value of debug_mode is: %d\n",debug_mode);
      }
   }

   int i1, i2, n;

   double **x = atom->x; // Get double pointer to atom positions
   int **bondlist = neighbor->bondlist;
   int nbondlist = neighbor->nbondlist;

   tagint *tag = atom->tag;
   
   
   // int ntimestep = update->ntimestep;

   /*
   if (rank == 0) {
      printf("The current timestep is: %d\n", ntimestep);
   }
   */


   int max_glo_molID;   
   // For each processor, store the maximum local molecule ID in max_glo_molID
   /*
   for (n = 0; n < nlocal; n++) {
      max_glo_molID = MAX(max_glo_molID, atom->molecule[n]);
   }
   */

   for (n = 0; n < nbondlist; n++) {
      i1 = bondlist[n][0];
      max_glo_molID = MAX(max_glo_molID, atom->molecule[i1]);
   }

   // We take the highest value of max_glo_molID across all processors, and
   // distribute that value to max_glo_molID to the rest of the processors
   // (hence the usage of MPI_IN_PLACE as the send buffer)
   MPI_Allreduce(MPI_IN_PLACE, &max_glo_molID, 1, MPI_INT, MPI_MAX, world);

   // We want to create a double-type array with max_glo_molID elements
   // to store the arc lengths of each molecule in the system
   double molLengths[max_glo_molID] = {0};

   double delx;
   double dely;
   double delz;

   double delxsq;
   double delysq;
   double delzsq;

   int mol1;
   int mol2;

   double dist;

   /*
   int size_Of_Cluster;
   MPI_Comm_size(world, &size_Of_Cluster);
   */

   if (debug_mode) {
      if (rank == 0) {
         printf("Begin calculation of end-to-end distances per molecule.\n");
      }
   }
   
   for (n = 0; n < nbondlist; n++) {
      dist = 0;
      i1 = bondlist[n][0]; // Get proc index of first atom in bond index n
      i2 = bondlist[n][1]; // Get proc index of second atom in bond index n

      // Each processor appears in the bondlist of one and only one proc
      // Therefore, we do not need to worry about under- or over- counting
      mol1 = atom->molecule[i1];
      mol2 = atom->molecule[i2];
      if (mol1 != mol2) error->all(FLERR, "There is a bond whose atoms belong to different molecules");

      delx = x[i1][0] - x[i2][0];
      dely = x[i1][1] - x[i2][1];
      delz = x[i1][2] - x[i2][2];

      delxsq = delx*delx;
      delysq = dely*dely;
      delzsq = delz*delz;

      // Calculate Euclidean distance of bond connecting atoms i1 and i2
      dist = sqrt(delxsq + delysq + delzsq);

      /* Use index mol1 - 1 because molecule ID indexing starts at 1
         but C++ indexing starts at 0 */
       molLengths[mol1 - 1] += dist;
      /* For half neighbor lists, because each bond is only stored once,
         there is no worry of double-counting a bond length */
     
      // Do not put printf statements that require substituting in values of LAMMPS variables inside of if statements 
      //   printf("Atom IDs (%d, %d) delx %f dely %f delz %f delxsq %f delysq %f delzsq %f dist %f\n", tag[i1], tag[i2], delx, dely, delz, delxsq, delysq, delzsq, dist);
      // printf("Atom IDs (%d, %d) x1 %f y1 %f z1 %f x2 %f y2 %f z2 %f\n", tag[i1], tag[i2], x[i1][0], x[i1][1], x[i1][2], x[i2][0], x[i2][1], x[i2][2]);
      
      /*
      if (isnan(molLengths[m - 1])) {
         printf("atom IDs (%d, %d) of molecule %d from process %d\n", tag[i1], tag[i2], m, rank);
      }
      */

   }
   
   /*
   for (n = 0; n < nmols; n++) {
      printf("Processor %d: the accumulated arc length of molecule with ID %d is %f\n",rank, n + 1, molLengths[n]);
   }
   */  


   // Take all of the locally processor-owned arc lengths of each molecule, and 
   // sum them up in place to get the total arc length for each molecule
   MPI_Allreduce(MPI_IN_PLACE, &molLengths, nmols, MPI_DOUBLE, MPI_SUM, world);
   
   if (debug_mode) {
   if (rank == 0) {
      for (n = 0; n < nmols; n++) {
         printf("The length of molecule with ID %d is %f\n\n", n + 1, molLengths[n]);    
      }
   }
   }
   


   erestraint = 0;
   for (n = 0; n < nmols; n++) {
      erestraint += k/2 * ((molLengths[n] * molLengths[n] / equiLength) - 2 * molLengths[n] + equiLength);
   }

   if (debug_mode) {
      if (rank == 0) {
         printf("The total energy of the system is: %f\n", erestraint);  
         printf("### Calculation of global restraint energy complete. ###\n");
      }
   }

   return erestraint;
}

/* ------------------------------------------------------------------------- */

void FixArcLengthRestraint::set_forces() {
   int newton_bond = force->newton_bond;

   int i1, i2, n, typ_i1, typ_i2;
   // double ebond, fbond;

   double **x = atom->x; // Get double pointer to atom positions
   double **f = atom->f; // Get double pointer to atom forces
   int *type = atom->type; // Get pointer to atom types
   int **bondlist = neighbor->bondlist;
   int nbondlist = neighbor->nbondlist;
   int nlocal = atom->nlocal;
   // int nghost = atom->nghost;

   tagint *tag = atom->tag;
   
   int rank;
   MPI_Comm_rank(world, &rank);

   int ntimestep = update->ntimestep;

   /*
   if (rank == 0) {
      printf("The current timestep is: %d\n", ntimestep);
   }
   */


   int max_glo_molID;   
   // For each processor, store the maximum local molecule ID in max_glo_molID
   /*
   for (n = 0; n < nlocal; n++) {
      max_glo_molID = MAX(max_glo_molID, atom->molecule[n]);
   }
   */

   for (n = 0; n < nbondlist; n++) {
      i1 = bondlist[n][0];
      max_glo_molID = MAX(max_glo_molID, atom->molecule[i1]);
   }

   // We take the highest value of max_glo_molID across all processors, and
   // distribute that value to max_glo_molID to the rest of the processors
   // (hence the usage of MPI_IN_PLACE as the send buffer)
   MPI_Allreduce(MPI_IN_PLACE, &max_glo_molID, 1, MPI_INT, MPI_MAX, world);

   // We want to create a double-type array with max_glo_molID elements
   // to store the arc lengths of each molecule in the system
   double molLengths[max_glo_molID] = {0};

   double delx;
   double dely;
   double delz;

   double delxsq;
   double delysq;
   double delzsq;

   int mol1;
   int mol2;

   double dist;

   /*
   int size_Of_Cluster;
   MPI_Comm_size(world, &size_Of_Cluster);
   */

   // printf("!!! Begin calculation of end-to-end distances per molecule.\n");
   
   for (n = 0; n < nbondlist; n++) {
      dist = 0;
      i1 = bondlist[n][0]; // Get proc index of first atom in bond index n
      i2 = bondlist[n][1]; // Get proc index of second atom in bond index n

      // Each processor appears in the bondlist of one and only one proc
      // Therefore, we do not need to worry about under- or over- counting
      mol1 = atom->molecule[i1];
      mol2 = atom->molecule[i2];
      if (mol1 != mol2) error->all(FLERR, "There is a bond whose atoms belong to different molecules");

      delx = x[i1][0] - x[i2][0];
      dely = x[i1][1] - x[i2][1];
      delz = x[i1][2] - x[i2][2];

      delxsq = delx*delx;
      delysq = dely*dely;
      delzsq = delz*delz;

      // Calculate Euclidean distance of bond connecting atoms i1 and i2
      dist = sqrt(delxsq + delysq + delzsq);

      /* Use index mol1 - 1 because molecule ID indexing starts at 1
         but C++ indexing starts at 0 */
       molLengths[mol1 - 1] += dist;
      /* For half neighbor lists, because each bond is only stored once,
         there is no worry of double-counting a bond length */

      /*
      if (isnan(molLengths[m - 1])) {
         printf("atom IDs (%d, %d) of molecule %d from process %d\n", tag[i1], tag[i2], m, rank);
      }
      */

   }
   
   /*
   for (n = 0; n < nmols; n++) {
      printf("Processor %d: the accumulated arc length of molecule with ID %d is %f\n",rank, n + 1, molLengths[n]);
   }
   */  


   // Take all of the locally processor-owned arc lengths of each molecule, and 
   // sum them up in place to get the total arc length for each molecule
   MPI_Allreduce(MPI_IN_PLACE, &molLengths, nmols, MPI_DOUBLE, MPI_SUM, world);
   
   if (debug_mode) { 
   if (rank == 0) {
      for (n = 0; n < nmols; n++) {
         printf("!!! The length of molecule with ID %d is %f\n\n", n + 1, molLengths[n]);    
      }
   }
   }


   double scalingFactors[max_glo_molID];
   for(n = 0; n < max_glo_molID; n++) {
      scalingFactors[n] = -1 * k * (molLengths[n] / equiLength - 1);
   }

   double scale;

   double fx_i1;
   double fy_i1;
   double fz_i1;

   int glob_id1;
   int glob_id2;

   if (debug_mode){
      if (rank == 0) {
         printf("!!!ALERT!!!: Begin validation of pre-assignment forces on system.\n");
         printf("Format:\n");
         printf(" id1 | id2 |  fx_i1 |  fy_i1 |  fz_i1 |  fx_i2 |  fy_i2 |  fz_i2 \n");  
      }

      double tot_fx = 0;
      double tot_fy = 0;
      double tot_fz = 0;

      double fx_i2 = 0;
      double fy_i2 = 0;
      double fz_i2 = 0;

      // int glob_id1 = 0;
      // int glob_id2 = 0;
      for (n = 0; n < nbondlist; n++) {
         i1 = bondlist[n][0];
	 i2 = bondlist[n][1];

	 glob_id1 = tag[i1];
	 glob_id2 = tag[i2];
         
	 fx_i1 = f[i1][0];
	 fy_i1 = f[i1][1];
	 fz_i1 = f[i1][2];

	 fx_i2 = f[i2][0];
         fy_i2 = f[i2][1];
	 fz_i2 = f[i2][2];

         tot_fx += fx_i1;
	 tot_fy += fy_i1;
	 tot_fz += fz_i1;

	 printf(" %3d | %3d |%8f|%8f|%8f|%8f|%8f|%8f\n", glob_id1, glob_id2, fx_i1, fy_i1, fz_i1, fx_i2, fy_i2, fz_i2); 

      }

      if (rank == 0) {
         printf("!!!ALERT!!!: Validation of pre-assignment forces on system complete.\n\n");

         printf("!!!ALERT!!!: We will now be printing out force calculation data.\n");
         printf("Format:\n");
         printf(" id1 | id2 |  scale  |   delx  |   dely  |   delz  |   dist  |  fx_i1  |  fy_i1  |  fz_i1  \n");  
      }

   }

   //Now that we have the molecule lengths, we can allocate forces
   for (n = 0; n < nbondlist; n++){
      i1 = bondlist[n][0];
      i2 = bondlist[n][1];

      mol1 = atom->molecule[i1];

      scale = scalingFactors[mol1 - 1];

      typ_i1 = type[i1];
      typ_i2 = type[i2];

      glob_id1 = tag[i1];
      glob_id2 = tag[i2];

      delx = x[i1][0] - x[i2][0];
      dely = x[i1][1] - x[i2][1];
      delz = x[i1][2] - x[i2][2];

      delxsq = delx*delx;
      delysq = dely*dely;
      delzsq = delz*delz;

      dist = sqrt(delxsq + delysq + delzsq);


      fx_i1 = 0;
      fy_i1 = 0;
      fz_i1 = 0;

      
      // This check is to avoid a division by zero error.
      if (dist > 10e-18) {
         fx_i1 = scale * delx / dist;
         fy_i1 = scale * dely / dist;
         fz_i1 = scale * delz / dist;
      }

      // This is for standardization, because we assume that
      // the global type IDs for the atom of a molecule 
      // are sequentially-ordered.
      if (typ_i1 > typ_i2) {
         fx_i1 *= -1;
         fy_i1 *= -1;
         fz_i1 *= -1;
      }

      if (newton_bond || i1 < nlocal) { 
         f[i1][0] += fx_i1;
         f[i1][1] += fy_i1;
         f[i1][2] += fz_i1;
      }

      // It turns out that in order to do a reverse-communication,
      // you need to have defined one in your file.
      // So I'm doing this to avoid having to manually define one.
      if (newton_bond || i2 < nlocal) {
         f[i2][0] -= fx_i1;
         f[i2][1] -= fy_i1;
         f[i2][2] -= fz_i1;
      }

      // printf("Atom IDs (%d, %d) fx_i1 %f fy_i1 %f fz_i1 %f\n", tag[i1], tag[i2], fx_i1, fy_i1, fz_i1);

      if (debug_mode) {
         printf(" %3d | %3d |%9f|%9f|%9f|%9f|%9f|%9f|%9f|%9f\n", glob_id1, glob_id2, scale, delx, dely, delz, dist, fx_i1, fy_i1, fz_i1); 
      }

      // printf("Atom IDs (%d, %d) fx_i1 %f fy_i1 %f fz_i1 %f fx_i2 %f fy_i2 %f fz_i2 %f\n", tag[i1], tag[i2], f[i1][0], f[i1][1], f[i1][2], f[i2][0], f[i2][1], f[i2][2]);
   }

   // Communicate accumulated forces on ghost atoms back to owning procs:
   //if (newton_bond == 1) comm->reverse_comm(this);

   // Confirm that forces on atoms really are what we desire:
   
   if (debug_mode) {
      if (rank == 0) {
         printf("\n!!!ALERT!!!: Printing of force calculations complete.\n\n"); 
         printf("!!!ALERT!!!: Begin validation of forces on system.\n");
         printf("Format:\n");
         printf(" id1 | id2 |  fx_i1  |  fy_i1  |  fz_i1  |  fx_i2  |  fy_i2  |  fz_i2  \n"); 
      }

      double tot_fx = 0;
      double tot_fy = 0;
      double tot_fz = 0;

      double fx_i2 = 0;
      double fy_i2 = 0;
      double fz_i2 = 0;

      // int glob_id1 = 0;
      // int glob_id2 = 0;
      for (n = 0; n < nbondlist; n++) {
         i1 = bondlist[n][0];
	 i2 = bondlist[n][1];

	 glob_id1 = tag[i1];
	 glob_id2 = tag[i2];
         
	 fx_i1 = f[i1][0];
	 fy_i1 = f[i1][1];
	 fz_i1 = f[i1][2];

	 fx_i2 = f[i2][0];
         fy_i2 = f[i2][1];
	 fz_i2 = f[i2][2];

         tot_fx += fx_i1;
	 tot_fy += fy_i1;
	 tot_fz += fz_i1;

	 // Check if the second atom on the bond is the terminus of the molecule
	 // We need to add those forces up as part of our total
	 if (type[i2] == napmol) {
            tot_fx += fx_i2;
	    tot_fy += fy_i2;
	    tot_fz += fz_i2;
         }

	 printf(" %3d | %3d |%9f|%9f|%9f|%9f|%9f|%9f\n", glob_id1, glob_id2, fx_i1, fy_i1, fz_i1, fx_i2, fy_i2, fz_i2); 

         // printf("Atom IDs (%d, %d) f[i1][0]: %f f[i1][1]: %f f[i1][2]: %f f[i2][0]: %f f[i2][1]: %f f[i2][2]: %f\n", glob_id1, glob_id2, fx_i1, fy_i1, fz_i1, fx_i2, fy_i2, fz_i2);
      }

      if (rank == 0){
         printf("!!!ALERT!!!: Total forces on system in along following coordinates: x %f y %f z %f\n", tot_fx, tot_fy, tot_fz);
      }

   }
   
}
