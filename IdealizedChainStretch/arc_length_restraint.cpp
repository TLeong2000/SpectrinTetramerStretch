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

#include "arc_length_restraint.h"

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

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixArcLengthRestraint::FixArcLengthRestraint(LAMMPS *lmp, int narg, char **arg) : 
   Fix(lmp, narg, arg) 
{
   if (narg < 6) error->all(FLERR, "Insufficient args for fix bondrestraintharmonic command.");
   imol = atom->find_molecule(arg[4]);
   k = utils::numeric(FLERR,arg[5],false,lmp);
   equiLength = utils::numeric(FLERR,arg[6],false,lmp);
   napmol = (atom->molecules[imol])->natoms;
   nbpmol = napmol - 1;
   natoms = atom->natoms;
   nmols = (atom->natoms) / napmol; // Number of molecules in the system
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
   double ebond, fbond;

   double **x = atom->x; // Get double pointer to atom positions
   double **f = atom->f; // Get double pointer to atom forces
   int *type = atom->type; // Get double pointer to atom types
   int **bondlist = neighbor->bondlist;
   int nbondlist = neighbor->nbondlist;
   int nlocal = atom->nlocal;
   int nghost = atom->nghost;
   int newton_bond = force->newton_bond;

   int max_glo_molID;
   int max_loc_molID = 0;
   for (n = 0; n < nlocal; n++) {
      max_loc_molID = MAX(max_loc_molID, atom->molecule[n]);
   }
   MPI_Allreduce(&max_loc_molID, &max_glo_molID, 1, MPI_INT, MPI_MAX, world);

   double molLengths[max_glo_molID];
   double delx;
   double dely;
   double delz;

   int m;

   for (n = 0; n <nbondlist; n++){
      i1 = bondlist[n][0]; // Get index of first atom in bond index n
      i2 = bondlist[n][1]; // Get index of second atom in bond index n
      if (i1 < nlocal) {
         m = atom->molecule[i1]; // Get molecule ID
      }
      if (i2 < nlocal) {
         m = atom->molecule[i2]; // Get molecule ID
      }

      delx = x[i1][0] - x[i2][0];
      dely = x[i1][1] - x[i2][1];
      delz = x[i1][2] - x[i2][2];

      if (newton_bond || i1 < nlocal) {
         /* Use index m - 1 because molecule ID indexing starts at 1
            but C++ indexing starts at 0 */
         molLengths[m - 1] += delx*delx + dely*dely + delz*delz;
         /* For half neighbor lists, because each bond is only stored once,
            there is no worry of double-counting a bond length */
      }
      
   }

   MPI_Allreduce(&molLengths, &molLengths, nmols, MPI_DOUBLE, MPI_SUM, world);

   erestraint = 0;
   for (n = 0; n < nmols; n++) {
      erestraint += k/2 * (molLengths[n] / equiLength - 1) * (molLengths[n] / equiLength - 1);
   }

   double scale;
   // std::array<double, max_glo_molID> scaling_factors = k * (molLengths / equiLength - 1);

   double dist = 0;
   // Now that we have the molecule lengths, we can allocate forces
   for (n = 0; n < nbondlist; n++) {
      i1 = bondlist[n][0]; // Get index of first atom in bond index n
      i2 = bondlist[n][1]; // Get index of second atom in bond index n
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

      dist = delx*delx + dely*dely + delz*delz;

      
      if (i1 < nlocal) {
         // typ_i1 = typ_i2 + 1 XOR typ_i2 - 1
         if (typ_i1 < typ_i2) { // i1 cannot be last atom of mol it belongs to
            f[i1][0] += scale * delx / dist;
            f[i1][1] += scale * dely / dist;
            f[i1][2] += scale * delz / dist;
         }
         else { // i1 cannot be first atom of mol it belongs to
            f[i1][0] -= scale * delx / dist;
            f[i1][1] -= scale * dely / dist;
            f[i1][2] -= scale * delz / dist;
         }
      }

      if (i2 < nlocal) {
         // typ_i2 = typ_i1 + 1 XOR typ_i1 - 1
         if (typ_i2 < typ_i1) { // i2 cannot be last atom of mol it belongs to
            f[i2][0] += scale * delx / dist;
            f[i2][1] += scale * dely / dist;
            f[i2][2] += scale * delz / dist;
         }
         else { // i2 cannot be first atom of mol it belongs to
            f[i2][0] -= scale * delx / dist;
            f[i2][1] -= scale * dely / dist;
            f[i2][2] -= scale * delz / dist;
         }
      }
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
