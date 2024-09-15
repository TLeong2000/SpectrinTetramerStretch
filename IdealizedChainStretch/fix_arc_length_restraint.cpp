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

#include "fix_arc_length_restraint.h"

#include "math.h"
#include "stdlib.h"
#include "atom.h"
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
   scalar_flag = 1;
   extscalar = 1;

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

   // equiLength is the equilibrium length that we want to restrain
   // the arc length of the chain to.
   equiLength = utils::numeric(FLERR,arg[5],false,lmp);

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
   int i1, i2, n, typ_i1, typ_i2;
   // double ebond, fbond;

   double **x = atom->x; // Get double pointer to atom positions
   double **f = atom->f; // Get double pointer to atom forces
   int *type = atom->type; // Get pointer to atom types
   int **bondlist = neighbor->bondlist;
   int nbondlist = neighbor->nbondlist;
   int nlocal = atom->nlocal;
   // int nghost = atom->nghost;
   int newton_bond = force->newton_bond;

   tagint *tag = atom->tag;
   
   int rank;
   MPI_Comm_rank(world, &rank);

   int ntimestep = update->ntimestep;

   if (rank == 0) {
      printf("The current timestep is: %d\n", ntimestep);
   }


   int max_glo_molID;   
   // For each processor, store the maximum local molecule ID in max_glo_molID
   for (n = 0; n < nlocal; n++) {
      max_glo_molID = MAX(max_glo_molID, atom->molecule[n]);
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

   int m;

   double dist;

   /*
   int size_Of_Cluster;
   MPI_Comm_size(world, &size_Of_Cluster);
   */
   
   

   for (n = 0; n <nbondlist; n++){
      dist = 0;
      i1 = bondlist[n][0]; // Get index of first atom in bond index n
      i2 = bondlist[n][1]; // Get index of second atom in bond index n
      if (i1 < nlocal) {
         m = atom->molecule[i1]; // Get molecule ID
      }
      if (i2 < nlocal) {
         m = atom->molecule[i2]; // Get molecule ID
      }
      
      // printf("Bond with atom IDs (%d, %d) of molecule %d from process %d of %d\n", tag[i1], tag[i2], m, rank, size_Of_Cluster);

      // Get components of position difference between atoms i1 and i2
      delx = x[i1][0] - x[i2][0];
      dely = x[i1][1] - x[i2][1];
      delz = x[i1][2] - x[i2][2];

      delxsq = delx*delx;
      delysq = dely*dely;
      delzsq = delz*delz;

      // Calculate Euclidean distance of bond connecting atoms i1 and i2
      dist = sqrt(delxsq + delysq + delzsq);
      
      

      // 
      // If newton_bond is on (which it is by default, unless one mentions
      // "newton off" in one's LAMMPS script), each atom in each bond
      // is mentioned once and only once across all processors' neighborlists.
      //
      // To be more specific, let atoms 1 and 2 be bonded with each other:
      // if atom 1 appears as subelement 0 of some element in the bondlist
      // of a processor, atom 2 will appear as subelement 1 of the same element
      // in that processor's bondlist.
      // If newton_bond is on, neither atom 1 nor atom 2 will appear as the 
      // subelement of any element of any other processor's bondlist.
      // If newton_bond is off, AND atom 2 is a ghost atom, then atom 2
      // will ALSO appear as subelement 0 of some element in the bondlist 
      // of the processor where atom 2 is local, and atom 1 will ALSO appear
      // as subelement 1 of the same element in the bondlist of the processor
      // where atom 2 is local.
      //
      if (newton_bond || i1 < nlocal) {
         /* Use index m - 1 because molecule ID indexing starts at 1
            but C++ indexing starts at 0 */
         molLengths[m - 1] += dist;
         /* For half neighbor lists, because each bond is only stored once,
            there is no worry of double-counting a bond length */
      }
      
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
   
   /*
   if (rank == 0) {
      for (n = 0; n < nmols; n++) {
         printf("The length of molecule with ID %d is %f\n", n + 1, molLengths[n]);    
      }
   }
   */


   erestraint = 0;
   for (n = 0; n < nmols; n++) {
      erestraint += k/2 * (molLengths[n] / equiLength - 1) * (molLengths[n] / equiLength - 1);
   }

   double scale;

   double fx_i1;
   double fy_i1;
   double fz_i1;

   double fx_i2;
   double fy_i2;
   double fz_i2;

   // Now that we have the molecule lengths, we can allocate forces
   for (n = 0; n < nbondlist; n++) {
      i1 = bondlist[n][0]; // Get index of first atom in bond index n
      // bondlist[n][0] will always be a local atom
      i2 = bondlist[n][1]; // Get index of second atom in bond index n
      // bondlist[n][1] may or may not be a local atom


      if (i1 < nlocal) {
         m = atom->molecule[i1]; // Get molecule ID
      }
      if (i2 < nlocal) {
         m = atom->molecule[i2]; // Get molecule ID
      }

      scale = k * (molLengths[m - 1] / equiLength - 1);
      // scale = scaling_factors[m - 1];

      typ_i1 = type[i1];
      typ_i2 = type[i2];

      delx = x[i1][0] - x[i2][0];
      dely = x[i1][1] - x[i2][1];
      delz = x[i1][2] - x[i2][2];

      delxsq = delx*delx;
      delysq = dely*dely;
      delzsq = delz*delz;

      dist = sqrt(delxsq + delysq + delzsq);

      // printf("Atom IDs (%d, %d) types (%d, %d) scale %f delx %f dely %f delz %f dist %f\n", tag[i1], tag[i2], typ_i1, typ_i2, scale, delx, dely, delz, dist);

      fx_i1 = 0;
      fy_i1 = 0;
      fz_i1 = 0;

      if (i1 < nlocal) {
         // typ_i1 = typ_i2 + 1 XOR typ_i2 - 1
         if (typ_i1 < typ_i2) { // i1 cannot be last atom of mol it belongs to
            // If the bond distance is equal to 0, we want to restraining force
            // to be zero to avoid a division by zero - which means that
            // we will not add or subtract any force from the atom
            if (dist > 10e-12) {
               fx_i1 += scale * delx / dist;
               fy_i1 += scale * dely / dist;
               fz_i1 += scale * delz / dist;

            }
            // printf("%d < %d\n", typ_i1, typ_i2);

         }
         else { // i1 cannot be first atom of mol it belongs to

            if (dist > 10e-12) {
               fx_i1 -= scale * delx / dist;
               fy_i1 -= scale * dely / dist;
               fz_i1 -= scale * delz / dist;
            }
            // printf("%d > %d\n", typ_i1, typ_i2);
            
         }
         f[i1][0] += fx_i1;
         f[i1][1] += fy_i1;
         f[i1][2] += fz_i1;

      }
      

      fx_i2 = 0;
      fy_i2 = 0;
      fz_i2 = 0;

      if (i2 < nlocal) {
         // typ_i2 = typ_i1 + 1 XOR typ_i1 - 1
         if (typ_i1 < typ_i2) { // i2 cannot be first atom of mol it belongs to

            if (dist > 10e-12) {
               fx_i2 -= scale * delx / dist;
               fy_i2 -= scale * dely / dist;
               fz_i2 -= scale * delz / dist;
            }
            // printf("%d < %d\n", typ_i1, typ_i2);
            
         }
         else { // i2 cannot be last atom of mol it belongs to

            if (dist > 10e-12) {
               fx_i2 += scale * delx / dist;
               fy_i2 += scale * dely / dist;
               fz_i2 += scale * delz / dist;
            }
            // printf("%d > %d\n", typ_i1, typ_i2);
            
         }

         f[i2][0] += fx_i2;
         f[i2][1] += fy_i2;
         f[i2][2] += fz_i2;

      }

      // printf("Atom IDs (%d, %d) fx_i1 %f fx_i2 %f fy_i1 %f fy_i2 %f fz_i1 %f fz_i2 %f\n", tag[i1], tag[i2], fx_i1, fx_i2, fy_i1, fy_i2, fz_i1, fz_i2);  
   }

}

/* ---------------------------------------------------------------------- */

void FixArcLengthRestraint::min_post_force(int vflag) {
   post_force(vflag);
}

/* ------------------------------------------------------------------------- */

double FixArcLengthRestraint::compute_scalar() {
   return erestraint;
}
