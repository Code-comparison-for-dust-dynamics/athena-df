//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos.cpp
//! \brief implements common functions in class EquationOfState

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "eos.hpp"

// Keep communicated conserved states consistent with the EOS result stored in a restart.
// Do not reconstruct primitives or change momentum in this pass.
void EquationOfState::ApplyHydroConservedFloors(AthenaArray<Real> &cons,
    int il, int iu, int jl, int ju, int kl, int ku) {
  if (!PreBoundaryFloorsSupported()) return;
#if NON_BAROTROPIC_EOS
  const Real internal_energy_floor = pressure_floor_/(gamma_ - 1.0);
#endif
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real &density = cons(IDN,k,j,i);
#if NON_BAROTROPIC_EOS
        Real &energy = cons(IEN,k,j,i);
        const HydroFloorState floor_state = HydroConservedFloorState(density,
            cons(IM1,k,j,i), cons(IM2,k,j,i), cons(IM3,k,j,i), energy,
            internal_energy_floor);
        density = floor_state.density;
        if (floor_state.energy_floored) energy = floor_state.energy_floor;
#else
        // Isothermal hydro has no conserved energy or pressure/gamma floor data.
        FloorConservedDensity(density, density_floor_);
#endif
      }
    }
  }
}
