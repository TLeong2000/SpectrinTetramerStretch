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
FixStyle(arclengthrestraint, FixArcLengthRestraint);
#else

#ifndef LMP_FIX_ARC_LENGTH_RESTRAINT_H
#define LMP_FIX_ARC_LENGTH_RESTRAINT_H

#include "fix.h"

namespace LAMMPS_NS
{
    class FixArcLengthRestraint : public Fix
    {
        public:
        FixArcLengthRestraint(class LAMMPS *, int, char **);
        int setmask() override;
	void init() override;
	void min_setup(int) override;
        void post_force(int) override;
        void min_post_force(int) override;
	double compute_scalar() override;

        protected:
        int imol;
        double k, equiLength;
        int napmol, nbpmol, natoms, nmols;

        private:
        double erestraint;
	bool is_minimize;
        bool debug_mode;
	bool is_safe;

        void set_forces();

    };
}

#endif
#endif
