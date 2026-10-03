//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk_VSI.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cfloat>     // FLT_MIN
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp" // ran2()

#if (!NON_BAROTROPIC_EOS)
#error "This problem generator requires NON_BAROTROPIC_EOS!"
#endif
//#define COROT_WITH_PL // RL: added for sims in co-rotating frames 

namespace {

void GetCylCoordInSpherical(Coordinates *pco, Real &rad, Real &phi, Real &z,
                            int i, int j, int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
                        const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_NSH_rad_dust(const Real rad,   const Real phi, const Real z,
                                const Real ratio, const Real St,  const Real eta);
Real VelProfileCyl_NSH_phi_dust(const Real rad,   const Real phi, const Real z,
                                const Real ratio, const Real St,  const Real eta);
Real VelProfileCyl_NSH_phi_gas(const Real rad,   const Real phi, const Real z,
                               const Real ratio, const Real St,  const Real eta);
Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS];

Real gm0, r0, rho0, pvalue, cs2_0, qvalue, beta, alpha_vis, dfloor, dffloor, dust_percent_floor, Omega0,
time_drag, time_refine, refine_theta_upper, refine_theta_lower, refine_r_min, refine_r_max,
x1min, x1max, x2min, x2max, damping_rate, radius_inner_damping, radius_outer_damping,
theta_upper_damping, theta_lower_damping, upper_altitude_damping, lower_altitude_damping,
inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping;

Real rad_soft, gmp, inv_sqrt2gmp, rad_planet, phi_planet_0, z_planet,
gas_percent_floor, t0_planet, vk_planet, omega_planet, inv_omega_planet,
cs_planet, dslope, pslope, gMth, Hill_radius,
t_planet_growth, user_dt, A_gap;

bool Inner_Gas_Damping_Flag, Outer_Gas_Damping_Flag, Theta_Gas_Damping_Flag,
Inner_Dust_Damping_Flag, Outer_Dust_Damping_Flag, Isothermal_Flag, RadiativeConduction_Flag,
MassTransfer_Flag, TransferFeedback_Flag;

int PlanetaryGravityOrder;
std::string basename;

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

// User-defined Stopping time
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
// User-defined dust diffusivity
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
      int is, int ie, int js, int je, int ks, int ke);
// User-defined condutivity
void RadiativeCondution(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke);
// User-defined history
Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout);
// Adaptive Mesh Refinement Condition
int RefinementCondition(MeshBlock *pmb);
void Vr_outflow(const Real r_ac, const Real r_gh, const Real rho_ac,
                const Real rho_gh, const Real vr_ac, Real &vr_gh);
void Vr_Mdot(const Real r_ac, const Real r_gh, const Real rho_ac,
             const Real rho_gh, const Real vr_ac, Real &vr_gh);

Real MyMeshSpacingX1(Real x, RegionSize rs);
}
// namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void AccumulateData(MeshBlock *pmb);
void DoOutput(MeshBlock *pmb, int dlevel);
int IsBigEndianOutput();

void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void InnerWaveDampingGas(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons);
void OuterWaveDampingGas(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons);
void UpperWaveDampingGas(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons);
void LowerWaveDampingGas(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons);
void InnerWaveDampingDust(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons_df);
void OuterWaveDampingDust(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons_df);
void PinDustBeforeRelease(MeshBlock *pmb);
void PostUpdate(MeshBlock *pmb, const Real t_start, const Real t_end,
    const Real dt, const int stage, const bool final_stage, const bool final_main_stage);
void DustDensityPercentFloor(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
      AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
    std::stringstream msg;
    msg << "This problem file must be setup in the spherical_polar coordinate!" << std::endl;
    ATHENA_ERROR(msg);
  }

  std::string basename;
  basename = pin->GetString("job", "problem_id");

  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem", "GM", 0.0);
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);

  Isothermal_Flag          = pin->GetBoolean("problem",      "Isothermal_Flag");
  RadiativeConduction_Flag = pin->GetOrAddBoolean("problem", "RadiativeConduction_Flag", false);
  Inner_Gas_Damping_Flag   = pin->GetOrAddBoolean("problem", "Inner_Gas_Damping_Flag",   false);
  Outer_Gas_Damping_Flag   = pin->GetOrAddBoolean("problem", "Outer_Gas_Damping_Flag",   false);
  Theta_Gas_Damping_Flag   = pin->GetOrAddBoolean("problem", "Theta_Gas_Damping_Flag",   false);
  Inner_Dust_Damping_Flag  = pin->GetOrAddBoolean("problem", "Inner_Dust_Damping_Flag",  false);
  Outer_Dust_Damping_Flag  = pin->GetOrAddBoolean("problem", "Outer_Dust_Damping_Flag",  false);

  // Get parameters for initial density and velocity
  rho0   = pin->GetReal("problem",      "rho0");
  pvalue = pin->GetOrAddReal("problem", "pvalue", -1.0);

  // Get parameters of initial pressure and cooling parameters
  cs2_0  = pin->GetOrAddReal("problem", "cs2_0",  0.0025);
  qvalue = pin->GetOrAddReal("problem", "qvalue", -0.5);
  beta   = pin->GetOrAddReal("problem", "beta",   0.0);
  if (beta < 0.0) beta = 0.0;

  Real float_min = std::numeric_limits<float>::min();
  dfloor   = pin->GetOrAddReal("hydro",             "dfloor",   (1024*(float_min)));
  alpha_vis = pin->GetOrAddReal("problem",           "alpha_vis", 0.0);
  dffloor  = pin->GetOrAddReal("dust",              "dffloor",  (1024*(float_min)));
  Omega0   = pin->GetOrAddReal("orbital_advection", "Omega0",   0.0);
  dust_percent_floor = pin->GetOrAddReal("dust", "dust_percent_floor", 0.0);

  /*
  if (Omega0 != 0.0) {
    std::stringstream msg;
    msg << "In this disk_planet_dust_spherical.cpp, Omega0 must be equaled to 0!" << std::endl;
    ATHENA_ERROR(msg);
  } //*/

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    AllocateRealUserMeshDataField(1);
    ruser_mesh_data[0].NewAthenaArray(NDUSTVARS);
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
    }
  }

  time_drag   = pin->GetOrAddReal("dust",    "time_drag",   0.0);
  time_refine = pin->GetOrAddReal("problem", "time_refine", time_drag);       // not used here
  Globals::local_ndustfluids_inte = NDUSTFLUIDS;
  // Legacy dust/duty_cycle_b4time_drag inputs are accepted but no longer used.

  // The parameters of one planet
  rad_planet   = pin->GetOrAddReal("problem",  "rad_planet", 1.0);            // radial position of the planet
  phi_planet_0 = pin->GetOrAddReal("problem",  "phi_planet", M_PI);           // azimuthal position of the planet
  z_planet     = pin->GetOrAddReal("problem",  "z_planet",   0.0);            // vertical position of the planet
  t0_planet    = (pin->GetOrAddReal("problem", "t0_planet",  0.0))*TWO_PI*r0; // time to put in the planet
  gmp          = pin->GetOrAddReal("problem",  "GMp",        0.0);            // GM of the planet
  user_dt      = pin->GetOrAddReal("problem",  "user_dt",    0.0);

  vk_planet        = std::sqrt(gm0/rad_planet);
  omega_planet     = vk_planet/rad_planet;
  inv_omega_planet = 1.0/omega_planet;

  PlanetaryGravityOrder = pin->GetOrAddInteger("problem", "PlanetaryGravityOrder", 2);
  if ((PlanetaryGravityOrder != 2) && (PlanetaryGravityOrder != 4))
    PlanetaryGravityOrder = 2;

  cs_planet = std::sqrt(cs2_0*std::pow(rad_planet/r0, pslope));

  if (gmp != 0.0) inv_sqrt2gmp = 1.0/std::sqrt(2.0*gmp);
  gMth = gm0*cs_planet*cs_planet*cs_planet/(vk_planet*vk_planet*vk_planet);

  t_planet_growth  = pin->GetOrAddReal("problem", "t_planet_growth", 0.0)*TWO_PI*r0; // orbital number to grow the planet
  t_planet_growth *= inv_omega_planet*(gmp/gMth);
  t_planet_growth += t0_planet;

  Hill_radius = (std::pow(gmp/gm0*ONE_3RD, ONE_3RD)*rad_planet);

  rad_soft  = pin->GetOrAddReal("problem", "rs", 0.6); // softening length of the gravitational potential of planets
  rad_soft *= Hill_radius;

  //accretion_radius  = pin->GetOrAddReal("problem", "accretion_radius", 0.3); // Accretion radius of planets
  //accretion_radius *= Hill_radius;

  //if (accretion_radius < rad_soft) accretion_radius = 1.1*rad_soft;

  //accretion_rate = pin->GetOrAddReal("problem", "accretion_rate", 0.1); // Accretion radius of planets

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  x2min = pin->GetReal("mesh", "x2min");
  x2max = pin->GetReal("mesh", "x2max");

  refine_theta_lower = pin->GetOrAddReal("problem", "refine_theta_lower", 0.5*PI - std::sqrt(cs2_0));
  refine_theta_upper = pin->GetOrAddReal("problem", "refine_theta_upper", 0.5*PI + std::sqrt(cs2_0));

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.2);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  upper_altitude_damping = 1.5*std::sqrt(cs2_0);
  lower_altitude_damping = 1.5*std::sqrt(cs2_0);

  theta_upper_damping = x2min + upper_altitude_damping;
  theta_lower_damping = x2max - upper_altitude_damping;

  refine_r_min = pin->GetOrAddReal("problem", "refine_r_min", radius_inner_damping);
  refine_r_max = pin->GetOrAddReal("problem", "refine_r_max", radius_outer_damping);

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // Enroll userdef mesh, x1
  if (pin->GetOrAddReal("mesh", "x1rat", 1.0) < 0.0) {
    EnrollUserMeshGenerator(X1DIR, MyMeshSpacingX1);
  }

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }

  if (mesh_bcs[BoundaryFace::inner_x3] != GetBoundaryFlag("periodic")) {
    std::stringstream msg;
    msg << "The boundary condition in x3 direction must be periodic!" << std::endl;
    ATHENA_ERROR(msg);
  }

  if (mesh_bcs[BoundaryFace::outer_x3] != GetBoundaryFlag("periodic")) {
    std::stringstream msg;
    msg << "The boundary condition in x3 direction must be periodic!" << std::endl;
    ATHENA_ERROR(msg);
  }

  if (NDUSTFLUIDS > 0) {
    // Enroll user-defined dust stopping time
    EnrollUserDustStoppingTime(MyStoppingTime);

    // Enroll user-defined dust diffusivity
    EnrollDustDiffusivity(MyDustDiffusivity);

    // Enroll user-defined history file
    AllocateUserHistoryOutput(1);
    EnrollUserHistoryOutput(0, DustFluidsRatioMaximum, "RatioMax", UserHistoryOperation::max);
  }

  // Enroll Thermal Relaxation
  EnrollUserExplicitSourceFunction(MySource);
  // Pin dust to the gas before release, after every update and before boundary exchange.
  EnrollUserPostUpdateFunction(PostUpdate);

  // Enroll user-defined thermal conduction
  if ((!Isothermal_Flag) && (beta > 0.0) && (RadiativeConduction_Flag))
    EnrollConductionCoefficient(RadiativeCondution);

  // Enroll user-defined AMR criterion
  if (adaptive)
    EnrollUserRefinementCondition(RefinementCondition);

  return;
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  AllocateRealUserMeshBlockDataField(4);
  Real orb_defined;
  (porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  int dk = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  ruser_meshblock_data[0].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
  ruser_meshblock_data[1].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
  ruser_meshblock_data[2].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
  //ruser_meshblock_data[3].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
  //ruser_meshblock_data[4].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);

  for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
    Real x2 = pcoord->x2v(j);
#pragma omp simd
    for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
      Real x1 = pcoord->x1v(i);

      Real rad, phi, z;
      GetCylCoordInSpherical(pcoord, rad, phi, z, i, j, 0);

      Real cs_square    = PoverRho(rad, phi, z);
      Real vel_K_square = gm0/(rad);
      Real vel_K_sph    = std::sqrt(vel_K_square);
      Real eta_gas      = 0.5*(qvalue + pvalue)*cs_square/vel_K_square;

      ruser_meshblock_data[0](j, i) = DenProfileCyl_gas(rad, phi, z);
      ruser_meshblock_data[1](j, i) = PoverRho(rad, phi, z);
      ruser_meshblock_data[2](j, i) = VelProfileCyl_gas(rad, phi, z) - orb_defined*vK(porb, x1, x2, 0);
      //ruser_meshblock_data[3](j, i) = VelProfileCyl_NSH_rad_dust(rad, phi, z, initial_D2G[0], Stokes_number[0], eta_gas);
      //ruser_meshblock_data[4](j, i) = vel_K_sph + VelProfileCyl_NSH_phi_dust(rad, phi, z, initial_D2G[0], Stokes_number[0], eta_gas) - orb_defined*vK(porb, x1, x2, 0);
    }
	}


#ifdef COROT_WITH_PL
  /* RL: calculate the distance^2 of each cell to the planet
   * cos_jj_sph = cos(angle between two vectors, r and r_planet) in sph coor
   */
  ruser_meshblock_data[3].NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
  Real x1, x2, x3, cos_jj_sph, dcos_jj_sph_dtheta, dcos_jj_sph_dphi_over_sinx2;
  Real r_comp_sph[3] = {1.0, M_PI_2, (omega_planet - Omega0) * pmy_mesh->time + phi_planet_0};
  if (Globals::my_rank == 0 && gid == 0) {
    std::cout << "[P0]: sanity check r_comp_sph={" << r_comp_sph[0] << ", " << r_comp_sph[1]/M_PI << "π, " << r_comp_sph[2]/M_PI << "π}" << std::endl;
  }

  // pre-calculate distance^2
  for (int k=ks-NGHOST; k<=ke+NGHOST; k++) {
    x3 = pcoord->x3v(k);
    for (int j=js-NGHOST; j<=je+NGHOST; j++) {
      x2 = pcoord->x2v(j);
      for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
        x1 = pcoord->x1v(i);
        // (x1, x2, x3) = (r, theta, phi)

        cos_jj_sph = std::sin(x2)*std::sin(r_comp_sph[1])*std::cos(x3 - r_comp_sph[2]) + std::cos(x2)*std::cos(r_comp_sph[1]);
        dcos_jj_sph_dtheta = std::cos(x2)*std::sin(r_comp_sph[1])*std::cos(x3 - r_comp_sph[2]) - std::sin(x2)*std::cos(r_comp_sph[1]);
        dcos_jj_sph_dphi_over_sinx2 = std::sin(r_comp_sph[1]) * (-std::sin(x3 - r_comp_sph[2]));
        ruser_meshblock_data[3](k, j, i) = r_comp_sph[0]*r_comp_sph[0] + x1*x1 - 2*r_comp_sph[0]*x1 * cos_jj_sph;

      }
    }
  }
#endif /* COROT_WITH_PL */

}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  std::int64_t iseed = -1 - gid;
  Real gamma = peos->GetGamma();
  Real igm1  = 1.0/(gamma - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

  Real orb_defined;
  (porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);

        Real rad, phi, z;
        GetCylCoordInSpherical(pcoord, rad, phi, z, i, j, k);

        Real cs_square   = PoverRho(rad, phi, z);
        Real omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
        Real vel_K       = vK(porb, x1, x2, x3);
        Real vis_vel_cyl = -1.5*(alpha_vis*gamma*cs_square/rad/omega_dyn);
        Real vel_K_sph   = vel_K;

        Real den_gas        = DenProfileCyl_gas(rad, phi, z);
        Real vis_vel_r      = vis_vel_cyl*std::sin(x2);
        Real vel_gas_phi    = VelProfileCyl_gas(rad, phi, z);
        vel_gas_phi        -= orb_defined*vK(porb, x1, x2, x3);
        Real vel_gas_theta  = vis_vel_cyl*std::cos(x2);

        Real eta_gas = 0.5*(qvalue + pvalue)*cs_square/SQR(rad*omega_dyn);

        phydro->u(IDN, k, j, i) = den_gas;
        phydro->u(IM1, k, j, i) = den_gas*vis_vel_r;
        phydro->u(IM2, k, j, i) = den_gas*vel_gas_theta;
        phydro->u(IM3, k, j, i) = den_gas*vel_gas_phi;

        phydro->u(IEN, k, j, i)  = cs_square*phydro->u(IDN, k, j, i)*igm1;
        phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                      + SQR(phydro->u(IM3, k, j, i)))/den_gas;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real den_dust        = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
            //Real den_dust        = 1e-6*den_gas;
            Real vel_dust_cyl    = VelProfileCyl_NSH_rad_dust(rad, phi, z, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            Real vel_dust_phi    = vel_K_sph + VelProfileCyl_NSH_phi_dust(rad, phi, z, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            vel_dust_phi        -= orb_defined*vK(porb, x1, x2, x3);
            Real vel_dust_r      = vel_dust_cyl*std::sin(x2);
            Real vel_dust_theta  = vel_dust_cyl*std::cos(x2);

            pdustfluids->df_u(rho_id, k, j, i) = den_dust;
            pdustfluids->df_u(v1_id,  k, j, i) = den_dust*vel_dust_r;
            pdustfluids->df_u(v2_id,  k, j, i) = den_dust*vel_dust_theta;
            pdustfluids->df_u(v3_id,  k, j, i) = den_dust*vel_dust_phi;
          }
        }

        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n) {
            pscalars->s(n, k, j, i) = DenProfileCyl_dust(rad, phi, z, 1., 0.1);
          }
        }
      }
    }
  }
  return;
}


namespace {
//----------------------------------------------------------------------------------------

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  if ((gmp > 0.0) && (time >= t0_planet))
    PlanetaryGravity(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  if ((!Isothermal_Flag) && (beta > 0.0))
    ThermalRelaxation(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {

  int nc1 = pmb->ncells1;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          //Constant Stokes number in disk problems
          Real &st_time = stopping_time(dust_id, k, j, i);
          GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
          st_time = Stokes_number[dust_id]*std::pow(rad_arr(i), 1.5)*inv_sqrt_gm0;
        }
      }
    }
  }
  return;
}


void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {

  int nc1 = pmb->ncells1;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);
  Real gamma = pmb->peos->GetGamma();

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

          const Real &gas_pre = w(IPR, k, j, i);
          const Real &gas_den = w(IDN, k, j, i);

          Real inv_Omega_K = std::pow(rad_arr(i), 1.5)*inv_sqrt_gm0;
          Real nu_gas      = alpha_vis*gamma*inv_Omega_K*gas_pre/gas_den;

          Real &diffusivity = nu_dust(dust_id, k, j, i);
          diffusivity       = nu_gas/(1.0 + SQR(Stokes_number[dust_id]));

          Real &soundspeed  = cs_dust(dust_id, k, j, i);
          soundspeed        = std::sqrt(diffusivity*inv_Omega_K);
        }
      }
    }
  }
  return;
}


void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real inv_beta  = 1.0/beta;
  //Real inv_beta = (beta > 1.0e-16) ? 1.0/beta : 1.0e16;
  Real gamma_gas = pmb->peos->GetGamma();
  Real igm1      = 1.0/(gamma_gas - 1.0);

  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoordInSpherical(pmb->pcoord, rad, phi, z, i, j, k);

        const Real &gas_rho = prim(IDN, k, j, i);
        const Real &gas_pre = prim(IPR, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real omega_dyn  = std::sqrt(gm0/(rad*rad*rad));
        Real inv_t_cool = omega_dyn*inv_beta;
        //Real cs_square_init = PoverRho(rad, phi, z);
        Real cs_square_init = pmb->ruser_meshblock_data[1](j, i);

        Real delta_erg  = (gas_pre - gas_rho*cs_square_init)*igm1*inv_t_cool*dt;
        gas_erg        -= delta_erg;
        //Real delta_erg  = (gas_pre - gas_rho*cs_square_init)*igm1*(1.0 - exp(-omega_dyn*inv_beta*dt));
        //gas_erg        -= delta_erg;
      }
    }
  }
  return;
}

// For now, it only works on signle planetary gravity at the mid-plane.
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real phi_planet_move    = omega_planet*time + phi_planet_0;
  Real planet_gm;
  int nc1 = pmb->ncells1;

  phi_planet_move -= Omega0*time;

  if (time >= t0_planet) {
    AthenaArray<Real> distance_square, x_dis, y_dis, z_dis, R_dis, phi_dis, theta_dis;
    AthenaArray<Real> R_arr, phi_arr, z_arr, acc_R, acc_phi, acc_z,
      indirect_acc_R, indirect_acc_phi, gravity, acc_rad_sph, acc_theta_sph;

    distance_square.NewAthenaArray(nc1);

    x_dis.NewAthenaArray(nc1);
    y_dis.NewAthenaArray(nc1);
    z_dis.NewAthenaArray(nc1);

    R_dis.NewAthenaArray(nc1);
    phi_dis.NewAthenaArray(nc1);
    theta_dis.NewAthenaArray(nc1);

    R_arr.NewAthenaArray(nc1);
    phi_arr.NewAthenaArray(nc1);
    z_arr.NewAthenaArray(nc1);

    acc_R.NewAthenaArray(nc1);
    acc_phi.NewAthenaArray(nc1);
    acc_z.NewAthenaArray(nc1);

    acc_rad_sph.NewAthenaArray(nc1);
    acc_theta_sph.NewAthenaArray(nc1);

    indirect_acc_R.NewAthenaArray(nc1);
    indirect_acc_phi.NewAthenaArray(nc1);
    gravity.NewAthenaArray(nc1);

    //(t_planet_growth > t0_planet) ? (planet_gm = gmp*std::sin(0.5*PI*std::min(time/t_planet_growth, 1.0))) : (planet_gm = gmp);
    planet_gm = gmp;
    //Real max_dist_diff = 0.0, reduced_max_dist_diff = 0.0;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real &x3 = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; ++j) {
        Real &x2 = pmb->pcoord->x2v(j);
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real &x1 = pmb->pcoord->x1v(i);

          GetCylCoordInSpherical(pmb->pcoord, R_arr(i), phi_arr(i), z_arr(i), i, j, k);

#ifndef COROT_WITH_PL
          x_dis(i) = R_arr(i)*std::cos(phi_arr(i)) - rad_planet*std::cos(phi_planet_move);
          y_dis(i) = R_arr(i)*std::sin(phi_arr(i)) - rad_planet*std::sin(phi_planet_move);
#endif /* COROT_WITH_PL */
          z_dis(i) = z_arr(i) - z_planet; // used below, not only in distance_square
          R_dis(i)   = R_arr(i) - rad_planet*std::cos(phi_arr(i) - phi_planet_move);
          phi_dis(i) = rad_planet*std::sin(phi_arr(i) - phi_planet_move);

#ifndef COROT_WITH_PL
          distance_square(i) = SQR(x_dis(i)) + SQR(y_dis(i)) + SQR(z_dis(i));
#else /* COROT_WITH_PL */
          // RL: checked to be consistent when in a co-rotating frame with the planet
          //if (std::abs(distance_square(i) - pmb->ruser_meshblock_data[3](k, j, i)) > max_dist_diff) {
          //  max_dist_diff = distance_square(i) - pmb->ruser_meshblock_data[3](k, j, i);
          //}
          distance_square(i) = pmb->ruser_meshblock_data[3](k, j, i);
#endif /* COROT_WITH_PL */

          indirect_acc_R(i)   =  planet_gm*std::cos(phi_arr(i) - phi_planet_move)/SQR(rad_planet);
          indirect_acc_phi(i) = -planet_gm*std::sin(phi_arr(i) - phi_planet_move)/SQR(rad_planet);

          gravity(i) = ((PlanetaryGravityOrder == 2)) ?
          (planet_gm/std::pow(distance_square(i)+SQR(rad_soft), 1.5)) :
          (planet_gm*(5.0*SQR(rad_soft)+2.0*distance_square(i))/(2.0*std::pow(SQR(rad_soft)+distance_square(i), 2.5)));

          acc_R(i)   = gravity(i)*R_dis(i)   + indirect_acc_R(i);
          acc_phi(i) = gravity(i)*phi_dis(i) + indirect_acc_phi(i);
          acc_z(i)   = gravity(i)*z_dis(i);

          Real sintheta = std::sin(x2);
          Real costheta = std::cos(x2);

          acc_rad_sph(i)   = acc_R(i)*sintheta + acc_z(i)*costheta;
          acc_theta_sph(i) = acc_R(i)*costheta - acc_z(i)*sintheta;

          const Real &gas_rho  = prim(IDN, k, j, i);
          const Real &gas_vel1 = prim(IM1, k, j, i);
          const Real &gas_vel2 = prim(IM2, k, j, i);
          const Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);

          Real delta_mom1 = -dt*gas_rho*acc_rad_sph(i);
          Real delta_mom2 = -dt*gas_rho*acc_theta_sph(i);
          Real delta_mom3 = -dt*gas_rho*acc_phi(i);

          gas_mom1 += delta_mom1;
          gas_mom2 += delta_mom2;
          gas_mom3 += delta_mom3;

          if (!Isothermal_Flag) {
            Real &gas_erg  = cons(IEN, k, j, i);
            gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
          }
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
#pragma omp simd
            for (int i=pmb->is; i<=pmb->ie; ++i) {
              const Real &dust_rho = prim_df(rho_id, k, j, i);

              Real &dust_mom1 = cons_df(v1_id, k, j, i);
              Real &dust_mom2 = cons_df(v2_id, k, j, i);
              Real &dust_mom3 = cons_df(v3_id, k, j, i);

              Real delta_dust_mom1 = -dt*dust_rho*acc_rad_sph(i);
              Real delta_dust_mom2 = -dt*dust_rho*acc_theta_sph(i);
              Real delta_dust_mom3 = -dt*dust_rho*acc_phi(i);

              dust_mom1 += delta_dust_mom1;
              dust_mom2 += delta_dust_mom2;
              dust_mom3 += delta_dust_mom3;
            }
          }
        }
      }
    }
    /* RL: checked to be consistent when in a co-rotating frame with the planet
#ifdef MPI_PARALLEL
    MPI_Reduce(&max_dist_diff, &reduced_max_dist_diff, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    if (Globals::my_rank == 0) {
      std::cout << "P[" << Globals::my_rank << "]: max(max_dist_diff)=" << reduced_max_dist_diff << std::endl;
    }
#else
    std::cout << "P[" << Globals::my_rank << "]: max_dist_diff=" << max_dist_diff << std::endl;
#endif
    //*/

  }
  return;
}


Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout) {

  Real ratio_maximum = 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js,
      je = pmb->je, ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> &df_w = pmb->pdustfluids->df_w;
  AthenaArray<Real> &w       = pmb->phydro->w;

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real &gas_rho  = w(IDN, k, j, i);
          Real &dust_rho = df_w(rho_id, k, j, i);
          ratio_maximum  = std::max(ratio_maximum, dust_rho/gas_rho);
        }
      }
    }
  }
  return ratio_maximum;
}


void RadiativeCondution(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke) {

  Real inv_beta  = 1.0/beta;
  Real gamma_gas = pmb->peos->GetGamma();
  Real igm1      = 1.0/(gamma_gas - 1.0);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoordInSpherical(pmb->pcoord, rad, phi, z, i, j, k);

        const Real &gas_rho = w(IDN, k, j, i);
        const Real &gas_pre = w(IPR, k, j, i);

        Real inv_omega_dyn   = std::sqrt((rad*rad*rad)/gm0);
        Real internal_erg    = gas_rho*gas_pre*igm1;
        Real kappa_radiative = internal_erg*inv_omega_dyn*inv_beta;

        phdif->kappa(HydroDiffusion::DiffProcess::aniso, k, j, i) = kappa_radiative;
      }
    }
  }
  return;
}


void GetCylCoordInSpherical(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
  phi = pco->x3v(k);
  z   = pco->x1v(i)*std::cos(pco->x2v(j));
  return;
}


Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real cs2    = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0, pvalue);
  Real dentem = denmid*std::exp(gm0/cs2*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den, dfloor);
}


Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real cs2    = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0, pvalue);
  Real dentem = denmid/H_ratio*std::exp(gm0/(SQR(H_ratio)*cs2)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den, dffloor);
}


Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = cs2_0*std::pow(rad/r0, qvalue);
  return poverr;
}


Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real cs2 = PoverRho(rad, phi, z);
  Real vel = (pvalue+qvalue)*cs2/(gm0/rad) + (1.0+qvalue) - qvalue*rad/std::sqrt(rad*rad+z*z);
  vel      = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real VelProfileCyl_NSH_rad_dust(const Real rad, const Real phi, const Real z,
                                const Real ratio, const Real St, const Real eta) {
  Real vel = (2.0*St/(SQR(St) + SQR(1.0 + ratio)))*eta*std::sqrt(gm0/rad);
  return vel;
}


Real VelProfileCyl_NSH_phi_dust(const Real rad, const Real phi, const Real z,
                                const Real ratio, const Real St, const Real eta) {
  //Real vel = (1.0 + (1.0 + ratio)*eta/(SQR(1.0 + ratio) + SQR(St)))*std::sqrt(gm0/rad);
  Real vel = (1.0 + ratio)*eta/(SQR(1.0 + ratio) + SQR(St))*std::sqrt(gm0/rad);
  return vel;
}

Real VelProfileCyl_NSH_phi_gas(const Real rad, const Real phi, const Real z,
                               const Real ratio, const Real St, const Real eta) {
  Real vel = (1.0 + ratio + SQR(St))*eta/(SQR(1.0 + ratio) + SQR(St))*std::sqrt(gm0/rad);
  return vel;
}


int RefinementCondition(MeshBlock *pmb) {
  Real time = pmb->pmy_mesh->time;

  bool time_flag = (time >= time_refine);

  if (time_flag) {
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      for (int j=pmb->js; j<=pmb->je; j++) {
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real &rad   = pmb->pcoord->x1v(i);
          Real &theta = pmb->pcoord->x2v(j);

          bool theta_lower = (theta >= refine_theta_lower);
          bool theta_upper = (theta <= refine_theta_upper);
          bool rad_min     = (rad >= refine_r_min);
          bool rad_max     = (rad <= refine_r_max);

          if (theta_lower && theta_upper && rad_min && rad_max)
            return 1;
        }
      }
    }
  }

  return 0;
}



void Vr_outflow(const Real r_ac, const Real r_gh, const Real rho_ac,
               const Real rho_gh, const Real vr_ac, Real &vr_gh) {

  vr_gh = (vr_ac <= 0.0) ? ((rho_ac*SQR(r_ac)*vr_ac)/(rho_gh*SQR(r_gh))) : 0.0;
  return;
}


void Vr_Mdot(const Real r_ac, const Real r_gh, const Real rho_ac,
               const Real rho_gh, const Real vr_ac, Real &vr_gh) {

  vr_gh = rho_ac*SQR(r_ac)*vr_ac/(rho_gh*SQR(r_gh));
  return;
}

Real MyMeshSpacingX1(Real x, RegionSize rs) {
  // log-spaced mesh
  Real tmp = x*(std::log10(rs.x1max)-std::log10(rs.x1min)) + std::log10(rs.x1min);

  return std::pow(10.0,tmp);
}

} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real gamma = pmb->peos->GetGamma();

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad_gh, phi_gh, z_gh;
        Real rad_ac, phi_ac, z_ac;
        GetCylCoordInSpherical(pco, rad_gh,  phi_gh,  z_gh,  il-i, j, k);
        GetCylCoordInSpherical(pco, rad_ac, phi_ac, z_ac, il, j, k);

        Real cs_square   = PoverRho(rad_gh, phi_gh, z_gh);
        Real omega_dyn   = std::sqrt(gm0/(rad_gh*rad_gh*rad_gh));
        Real vis_vel_cyl = -1.5*(alpha_vis*gamma*cs_square/rad_gh/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
        Real eta_gas     = 0.5*(qvalue + pvalue)*cs_square/SQR(rad_gh*omega_dyn);
        //Real vel_K_sph   = std::sqrt(gm0/(pco->x1v(il-i)));
        Real vel_K_sph   = vel_K;

        Real &gas_rho_gh  = prim(IDN, k, j, il-i);
        Real &gas_vel1_gh = prim(IM1, k, j, il-i);
        Real &gas_vel2_gh = prim(IM2, k, j, il-i);
        Real &gas_vel3_gh = prim(IM3, k, j, il-i);
        Real &gas_pres_gh = prim(IEN, k, j, il-i);

        Real &gas_rho_ac  = prim(IDN, k, j, il);
        Real &gas_vel1_ac = prim(IM1, k, j, il);
        Real &gas_vel2_ac = prim(IM2, k, j, il);
        Real &gas_vel3_ac = prim(IM3, k, j, il);
        Real &gas_pres_ac = prim(IEN, k, j, il);

        gas_rho_gh        = DenProfileCyl_gas(rad_gh, phi_gh, z_gh);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_gh, phi_gh, z_gh);
        vel_gas_phi      -= orb_defined*vel_K;

        gas_vel1_gh = vis_vel_cyl*std::sin(pco->x2v(j));
        //Vr_outflow(rad_ac, rad_gh, gas_rho_ac, gas_rho_gh, gas_vel1_ac, gas_vel1_gh);
        gas_vel2_gh = vis_vel_cyl*std::cos(pco->x2v(j));
        gas_vel3_gh = vel_gas_phi;
        gas_pres_gh = cs_square*gas_rho_gh;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh  = prim_df(rho_id, k, j, il-i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, j, il-i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, j, il-i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, j, il-i);

            dust_rho_gh          = DenProfileCyl_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = VelProfileCyl_NSH_rad_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            Real vel_dust_phi    = vel_K_sph + VelProfileCyl_NSH_phi_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(pco->x2v(j));
            Real vel_dust_theta  = vel_dust_cyl*std::cos(pco->x2v(j));

            dust_vel1_gh = vel_dust_r;
            dust_vel2_gh = vel_dust_theta;
            dust_vel3_gh = vel_dust_phi;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real gamma = pmb->peos->GetGamma();

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad_gh, phi_gh, z_gh;
        GetCylCoordInSpherical(pco, rad_gh, phi_gh, z_gh, iu+i, j, k);

        Real cs_square   = PoverRho(rad_gh, phi_gh, z_gh);
        Real omega_dyn   = std::sqrt(gm0/(rad_gh*rad_gh*rad_gh));
        Real vis_vel_cyl = -1.5*(alpha_vis*gamma*cs_square/rad_gh/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
        Real eta_gas     = 0.5*(qvalue + pvalue)*cs_square/SQR(rad_gh*omega_dyn);
        //Real vel_K_sph   = std::sqrt(gm0/(pco->x1v(iu+i)));
        Real vel_K_sph   = vel_K;

        Real &gas_rho_gh  = prim(IDN, k, j, iu+i);
        Real &gas_vel1_gh = prim(IM1, k, j, iu+i);
        Real &gas_vel2_gh = prim(IM2, k, j, iu+i);
        Real &gas_vel3_gh = prim(IM3, k, j, iu+i);
        Real &gas_pres_gh = prim(IEN, k, j, iu+i);

        gas_rho_gh        = DenProfileCyl_gas(rad_gh, phi_gh, z_gh);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_gh, phi_gh, z_gh);
        vel_gas_phi      -= orb_defined*vel_K;

        gas_vel1_gh = vis_vel_cyl*std::sin(pco->x2v(j));
        gas_vel2_gh = vis_vel_cyl*std::cos(pco->x2v(j));
        gas_vel3_gh = vel_gas_phi;
        gas_pres_gh = cs_square*gas_rho_gh;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh  = prim_df(rho_id, k, j, iu+i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, j, iu+i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, j, iu+i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, j, iu+i);

            dust_rho_gh          = DenProfileCyl_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = VelProfileCyl_NSH_rad_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            Real vel_dust_phi    = vel_K_sph + VelProfileCyl_NSH_phi_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(pco->x2v(j));
            Real vel_dust_theta  = vel_dust_cyl*std::cos(pco->x2v(j));

            dust_vel1_gh = vel_dust_r;
            dust_vel2_gh = vel_dust_theta;
            dust_vel3_gh = vel_dust_phi;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real gamma = pmb->peos->GetGamma();

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real rad_gh, phi_gh, z_gh;
        Real rad_ac, phi_ac, z_ac;
        GetCylCoordInSpherical(pco, rad_gh, phi_gh, z_gh, i, jl-j, k);
        GetCylCoordInSpherical(pco, rad_ac, phi_ac, z_ac, i, jl,   k);

        Real cs_square   = PoverRho(rad_gh, phi_gh, z_gh);
        Real omega_dyn   = std::sqrt(gm0/(rad_gh*rad_gh*rad_gh));
        Real vis_vel_cyl = -1.5*(alpha_vis*gamma*cs_square/rad_gh/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
        Real eta_gas     = 0.5*(qvalue + pvalue)*cs_square/SQR(rad_gh*omega_dyn);
        //Real vel_K_sph   = std::sqrt(gm0/(pco->x1v(i)));
        Real vel_K_sph   = vel_K;

        Real &gas_rho_gh  = prim(IDN, k, jl-j, i);
        Real &gas_vel1_gh = prim(IM1, k, jl-j, i);
        Real &gas_vel2_gh = prim(IM2, k, jl-j, i);
        Real &gas_vel3_gh = prim(IM3, k, jl-j, i);
        Real &gas_pres_gh = prim(IEN, k, jl-j, i);

        Real &gas_rho_ac  = prim(IDN, k, jl, i);
        Real &gas_vel1_ac = prim(IM1, k, jl, i);
        Real &gas_vel2_ac = prim(IM2, k, jl, i);
        Real &gas_vel3_ac = prim(IM3, k, jl, i);
        Real &gas_pres_ac = prim(IEN, k, jl, i);

        gas_rho_gh        = DenProfileCyl_gas(rad_gh, phi_gh, z_gh);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_gh, phi_gh, z_gh);
        vel_gas_phi      -= orb_defined*vel_K;

        gas_vel1_gh = vis_vel_cyl*std::sin(pco->x2v(jl-j));
        //gas_vel2_gh = gas_vel2_ac;
        //Vr_Mdot(rad_ac, rad_gh, gas_rho_ac, gas_rho_gh, gas_vel1_ac, gas_vel1_gh);
        gas_vel2_gh = vis_vel_cyl*std::cos(pco->x2v(jl-j));
        gas_vel3_gh = vel_gas_phi;
        gas_pres_gh = cs_square*gas_rho_gh;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh  = prim_df(rho_id, k, jl-j, i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, jl-j, i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, jl-j, i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, jl-j, i);

            dust_rho_gh          = DenProfileCyl_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = VelProfileCyl_NSH_rad_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            Real vel_dust_phi    = vel_K_sph + VelProfileCyl_NSH_phi_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(pco->x2v(jl-j));
            Real vel_dust_theta  = vel_dust_cyl*std::cos(pco->x2v(jl-j));

            dust_vel1_gh = vel_dust_r;
            dust_vel2_gh = vel_dust_theta;
            dust_vel3_gh = vel_dust_phi;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real gamma = pmb->peos->GetGamma();

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real rad_gh, phi_gh, z_gh;
        Real rad_ac, phi_ac, z_ac;
        GetCylCoordInSpherical(pco, rad_gh, phi_gh, z_gh, i, ju+j, k);
        GetCylCoordInSpherical(pco, rad_ac, phi_ac, z_ac, i, ju,   k);

        Real cs_square   = PoverRho(rad_gh, phi_gh, z_gh);
        Real omega_dyn   = std::sqrt(gm0/(rad_gh*rad_gh*rad_gh));
        Real vis_vel_cyl = -1.5*(alpha_vis*gamma*cs_square/rad_gh/omega_dyn);
        Real vel_K       = vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
        Real eta_gas     = 0.5*(qvalue + pvalue)*cs_square/SQR(rad_gh*omega_dyn);
        //Real vel_K_sph   = std::sqrt(gm0/(pco->x1v(i)));
        Real vel_K_sph   = vel_K;

        Real &gas_rho_gh  = prim(IDN, k, ju+j, i);
        Real &gas_vel1_gh = prim(IM1, k, ju+j, i);
        Real &gas_vel2_gh = prim(IM2, k, ju+j, i);
        Real &gas_vel3_gh = prim(IM3, k, ju+j, i);
        Real &gas_pres_gh = prim(IEN, k, ju+j, i);

        Real &gas_rho_ac  = prim(IDN, k, ju, i);
        Real &gas_vel1_ac = prim(IM1, k, ju, i);
        Real &gas_vel2_ac = prim(IM2, k, ju, i);
        Real &gas_vel3_ac = prim(IM3, k, ju, i);
        Real &gas_pres_ac = prim(IEN, k, ju, i);

        gas_rho_gh        = DenProfileCyl_gas(rad_gh, phi_gh, z_gh);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_gh, phi_gh, z_gh);
        vel_gas_phi      -= orb_defined*vel_K;

        gas_vel1_gh = vis_vel_cyl*std::sin(pco->x2v(ju+j));
        //gas_vel2_gh = gas_vel2_ac;
        //Vr_Mdot(rad_ac, rad_gh, gas_rho_ac, gas_rho_gh, gas_vel1_ac, gas_vel1_gh);
        gas_vel2_gh = vis_vel_cyl*std::cos(pco->x2v(ju+j));
        gas_vel3_gh = vel_gas_phi;
        gas_pres_gh = cs_square*gas_rho_gh;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh  = prim_df(rho_id, k, ju+j, i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, ju+j, i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, ju+j, i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, ju+j, i);

            dust_rho_gh          = DenProfileCyl_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_cyl    = VelProfileCyl_NSH_rad_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            Real vel_dust_phi    = vel_K_sph + VelProfileCyl_NSH_phi_dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(pco->x2v(ju+j));
            Real vel_dust_theta  = vel_dust_cyl*std::cos(pco->x2v(ju+j));

            dust_vel1_gh = vel_dust_r;
            dust_vel2_gh = vel_dust_theta;
            dust_vel3_gh = vel_dust_phi;
          }
        }
      }
    }
  }
}


void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        //Real rad, phi, z;
        //GetCylCoordInSpherical(pmb->pcoord, rad, phi, z, i, j, k);

        Real &gas_pres = prim(IPR, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real inv_gas_dens = 1.0/gas_dens;
        //gas_pres = PoverRho(rad, phi, z)*gas_dens;
        gas_pres = pmb->ruser_meshblock_data[1](j, i)*gas_dens;
        gas_erg  = gas_pres*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
      }
    }
  }
  return;
}


// The damping operators act on conserved variables in the post-update task, before the
// boundary exchange, so restarted and uninterrupted runs see the same ghost states.
inline void DampGasConservedCell(AthenaArray<Real> &cons, const int k, const int j,
    const int i, const Real rho_0, const Real vel1_0, const Real vel2_0,
    const Real vel3_0, const Real pressure_0, const Real damping_factor,
    const Real gm1, const Real density_floor, const Real pressure_floor,
    const Real internal_energy_floor) {
  Real &density = cons(IDN, k, j, i);
  Real &mom1 = cons(IM1, k, j, i);
  Real &mom2 = cons(IM2, k, j, i);
  Real &mom3 = cons(IM3, k, j, i);
  Real &energy = cons(IEN, k, j, i);

  // Reproduce the EOS floors that preceded the old UserWorkInLoop damping path.
  density = (density > density_floor) ? density : density_floor;
  Real inv_density = 1.0/density;
  Real vel1 = mom1*inv_density;
  Real vel2 = mom2*inv_density;
  Real vel3 = mom3*inv_density;
  Real kinetic = 0.5*inv_density*(SQR(mom1) + SQR(mom2) + SQR(mom3));
  Real pressure = gm1*(energy - kinetic);
  Real energy_floor = internal_energy_floor + kinetic;
  if (!(energy > energy_floor)) {
    pressure = pressure_floor;
    energy = energy_floor;
  }

  density += (rho_0 - density)*damping_factor;
  vel1 += (vel1_0 - vel1)*damping_factor;
  vel2 += (vel2_0 - vel2)*damping_factor;
  vel3 += (vel3_0 - vel3)*damping_factor;
  pressure += (pressure_0 - pressure)*damping_factor;

  mom1 = density*vel1;
  mom2 = density*vel2;
  mom3 = density*vel3;
  kinetic = 0.5*(SQR(mom1) + SQR(mom2) + SQR(mom3))/density;
  energy = pressure/gm1 + kinetic;
}

inline void DampDustConservedCell(MeshBlock *pmb, AthenaArray<Real> &cons_df,
    const int dust_id, const int k, const int j, const int i, const Real rho_0,
    const Real vel1_0, const Real vel2_0, const Real vel3_0,
    const Real damping_factor) {
  DustFluids *pdf = pmb->pdustfluids;
  int rho_id = 4*dust_id;
  int v1_id = rho_id + 1;
  int v2_id = rho_id + 2;
  int v3_id = rho_id + 3;

  Real &density = cons_df(rho_id, k, j, i);
  Real &mom1 = cons_df(v1_id, k, j, i);
  Real &mom2 = cons_df(v2_id, k, j, i);
  Real &mom3 = cons_df(v3_id, k, j, i);
  density = (density > pmb->peos->GetDustFluidsFloor(dust_id))
            ? density : pmb->peos->GetDustFluidsFloor(dust_id);

  // Conserved dust momentum includes the diffusion momentum. Damp the physical velocity
  // and keep the diffusion momentum unchanged until the diffusion operator recomputes it.
  const bool with_diff = pdf->dfdif.MomentumIncludesDiffusion();
  const Real diff_mom1 = with_diff ? pdf->dfccdif.diff_mom_cc(v1_id, k, j, i) : 0.0;
  const Real diff_mom2 = with_diff ? pdf->dfccdif.diff_mom_cc(v2_id, k, j, i) : 0.0;
  const Real diff_mom3 = with_diff ? pdf->dfccdif.diff_mom_cc(v3_id, k, j, i) : 0.0;
  Real inv_density = 1.0/density;
  Real vel1 = (mom1 - diff_mom1)*inv_density;
  Real vel2 = (mom2 - diff_mom2)*inv_density;
  Real vel3 = (mom3 - diff_mom3)*inv_density;

  density += (rho_0 - density)*damping_factor;
  vel1 += (vel1_0 - vel1)*damping_factor;
  vel2 += (vel2_0 - vel2)*damping_factor;
  vel3 += (vel3_0 - vel3)*damping_factor;

  mom1 = density*vel1 + diff_mom1;
  mom2 = density*vel2 + diff_mom2;
  mom3 = density*vel3 + diff_mom3;
}

void InnerWaveDampingGas(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons) {

  Real gamma = pmb->peos->GetGamma();
  Real gm1 = gamma - 1.0;
  Real density_floor = pmb->peos->GetDensityFloor();
  Real pressure_floor = pmb->peos->GetPressureFloor();
  Real internal_energy_floor = pressure_floor/gm1;

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) <= radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_cyl  = -1.5*(alpha_vis*gamma*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_cyl*std::sin(x2);
          Real gas_vel2_0 = vis_vel_cyl*std::cos(x2);
          Real gas_vel3_0 = vel_gas_phi;
          Real gas_pre_0  = cs_square_0*gas_rho_0;

          Real damping_factor = R_func(i)*inv_damping_tau(i)*dt;
          DampGasConservedCell(cons, k, j, i, gas_rho_0, gas_vel1_0,
                               gas_vel2_0, gas_vel3_0, gas_pre_0,
                               damping_factor, gm1, density_floor, pressure_floor,
                               internal_energy_floor);
        }
      }
    }
  }
  return;
}


void OuterWaveDampingGas(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons) {

  Real gamma = pmb->peos->GetGamma();
  Real gm1 = gamma - 1.0;
  Real density_floor = pmb->peos->GetDensityFloor();
  Real pressure_floor = pmb->peos->GetPressureFloor();
  Real internal_energy_floor = pressure_floor/gm1;

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) >= radius_outer_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_outer_damping)*inv_outer_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_cyl  = -1.5*(alpha_vis*gamma*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_cyl*std::sin(x2);
          Real gas_vel2_0 = vis_vel_cyl*std::cos(x2);
          Real gas_vel3_0 = vel_gas_phi;
          Real gas_pre_0  = cs_square_0*gas_rho_0;

          Real damping_factor = R_func(i)*inv_damping_tau(i)*dt;
          DampGasConservedCell(cons, k, j, i, gas_rho_0, gas_vel1_0,
                               gas_vel2_0, gas_vel3_0, gas_pre_0,
                               damping_factor, gm1, density_floor, pressure_floor,
                               internal_energy_floor);
        }
      }
    }
  }
  return;
}


void UpperWaveDampingGas(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons) {

  Real gamma = pmb->peos->GetGamma();
  Real gm1 = gamma - 1.0;
  Real density_floor = pmb->peos->GetDensityFloor();
  Real pressure_floor = pmb->peos->GetPressureFloor();
  Real internal_energy_floor = pressure_floor/gm1;

  Real inv_upper_damp = 1.0/upper_altitude_damping;
  Real inv_lower_damp = 1.0/lower_altitude_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, Theta_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  Theta_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (x2 <= theta_upper_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          Theta_func(i)      = SQR((x2 - theta_upper_damping)*inv_upper_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_cyl  = -1.5*(alpha_vis*gamma*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_cyl*std::sin(x2);
          Real gas_vel2_0 = vis_vel_cyl*std::cos(x2);
          Real gas_vel3_0 = vel_gas_phi;
          Real gas_pre_0  = cs_square_0*gas_rho_0;

          Real damping_factor = Theta_func(i)*inv_damping_tau(i)*dt;
          DampGasConservedCell(cons, k, j, i, gas_rho_0, gas_vel1_0,
                               gas_vel2_0, gas_vel3_0, gas_pre_0,
                               damping_factor, gm1, density_floor, pressure_floor,
                               internal_energy_floor);
        }
      }
    }
  }
  return;
}


void LowerWaveDampingGas(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons) {

  Real gamma = pmb->peos->GetGamma();
  Real gm1 = gamma - 1.0;
  Real density_floor = pmb->peos->GetDensityFloor();
  Real pressure_floor = pmb->peos->GetPressureFloor();
  Real internal_energy_floor = pressure_floor/gm1;

  Real inv_upper_damp = 1.0/upper_altitude_damping;
  Real inv_lower_damp = 1.0/lower_altitude_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, Theta_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  Theta_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (x2 >= theta_lower_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          Theta_func(i)      = SQR((x2 - theta_lower_damping)*inv_lower_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_0   = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));

          Real vis_vel_cyl  = -1.5*(alpha_vis*gamma*cs_square_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0 = vis_vel_cyl*std::sin(x2);
          Real gas_vel2_0 = vis_vel_cyl*std::cos(x2);
          Real gas_vel3_0 = vel_gas_phi;
          Real gas_pre_0  = cs_square_0*gas_rho_0;

          Real damping_factor = Theta_func(i)*inv_damping_tau(i)*dt;
          DampGasConservedCell(cons, k, j, i, gas_rho_0, gas_vel1_0,
                               gas_vel2_0, gas_vel3_0, gas_pre_0,
                               damping_factor, gm1, density_floor, pressure_floor,
                               internal_energy_floor);
        }
      }
    }
  }
  return;
}


void InnerWaveDampingDust(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons_df) {

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

          if (rad_arr(i) <= radius_inner_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
            R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
            Real eta_gas     = 0.5*(qvalue + pvalue)*cs_square_0/SQR(rad_arr(i)*omega_dyn(i));
            Real vel_K           = vK(pmb->porb, x1, x2, x3);
            Real vel_dust_cyl    = VelProfileCyl_NSH_rad_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            Real vel_dust_phi    = vel_K + VelProfileCyl_NSH_phi_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(x2);
            Real vel_dust_theta  = vel_dust_cyl*std::cos(x2);

            Real dust_rho_0  = DenProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Hratio[dust_id]);
            Real dust_vel1_0 = vel_dust_r;
            Real dust_vel2_0 = vel_dust_theta;
            Real dust_vel3_0 = vel_dust_phi;

            Real damping_factor = R_func(i)*inv_damping_tau(i)*dt;
            DampDustConservedCell(pmb, cons_df, dust_id, k, j, i, dust_rho_0,
                                  dust_vel1_0, dust_vel2_0, dust_vel3_0,
                                  damping_factor);
          }
        }
      }
    }
  }
  return;
}


void OuterWaveDampingDust(MeshBlock *pmb, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &cons_df) {

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

          if (rad_arr(i) >= radius_outer_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
            R_func(i)          = SQR((rad_arr(i) - radius_outer_damping)*inv_outer_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs_square_0 = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
            Real eta_gas     = 0.5*(qvalue + pvalue)*cs_square_0/SQR(rad_arr(i)*omega_dyn(i));
            Real vel_K           = vK(pmb->porb, x1, x2, x3);
            Real vel_dust_cyl    = VelProfileCyl_NSH_rad_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            Real vel_dust_phi    = vel_K + VelProfileCyl_NSH_phi_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Stokes_number[dust_id], eta_gas);
            vel_dust_phi        -= orb_defined*vel_K;
            Real vel_dust_r      = vel_dust_cyl*std::sin(x2);
            Real vel_dust_theta  = vel_dust_cyl*std::cos(x2);

            Real dust_rho_0  = DenProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Hratio[dust_id]);
            Real dust_vel1_0 = vel_dust_r;
            Real dust_vel2_0 = vel_dust_theta;
            Real dust_vel3_0 = vel_dust_phi;

            Real damping_factor = R_func(i)*inv_damping_tau(i)*dt;
            DampDustConservedCell(pmb, cons_df, dust_id, k, j, i, dust_rho_0,
                                  dust_vel1_0, dust_vel2_0, dust_vel3_0,
                                  damping_factor);
          }
        }
      }
    }
  }
  return;
}


// Pin only conserved active cells; normal stage communication and cons-to-prim
// fill ghosts/primitives exactly as restart initialization does. No diffusive
// momentum is retained while dust follows the gas before release.
void PinDustBeforeRelease(MeshBlock *pmb) {
  if (NDUSTFLUIDS <= 0) return;
  const Real t_release = pmb->pdustfluids->dfdrag.time_drag;
  if ((t_release <= 0.0) || (pmb->pmy_mesh->time >= t_release)) return;
  const AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> &cons_df = pmb->pdustfluids->df_u;
  for (int n=0; n<NDUSTFLUIDS; ++n) {
    const int rho_id = 4*n;
    const Real d2g = initial_D2G[n];
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          cons_df(rho_id,   k, j, i) = d2g*u(IDN, k, j, i);
          cons_df(rho_id+1, k, j, i) = d2g*u(IM1, k, j, i);
          cons_df(rho_id+2, k, j, i) = d2g*u(IM2, k, j, i);
          cons_df(rho_id+3, k, j, i) = d2g*u(IM3, k, j, i);
        }
      }
    }
  }
  // no diffusive momentum while dust is pinned (dust diffusion is not computed)
  if (pmb->pdustfluids->dfdif.dustfluids_diffusion_defined)
    pmb->pdustfluids->dfccdif.diff_mom_cc.ZeroClear();
}

void PostUpdate(MeshBlock *pmb, const Real t_start, const Real t_end,
    const Real dt, const int stage, const bool final_stage, const bool final_main_stage) {
  (void)t_start; (void)t_end; (void)dt; (void)stage; (void)final_stage;
  // Damping must precede the final OA-only stage. That stage performs the last standard
  // boundary exchange/prolongation and leaves restart-rebuilt SMR ghost states identical to
  // uninterrupted evolution. It acts once per cycle with the cycle timestep.
  if (final_main_stage) {
    Real cycle_dt = pmb->pmy_mesh->dt;
    int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je, kl = pmb->ks, ku = pmb->ke;
    if (Inner_Gas_Damping_Flag)
      InnerWaveDampingGas(pmb, cycle_dt, il, iu, jl, ju, kl, ku, pmb->phydro->u);
    if (Outer_Gas_Damping_Flag)
      OuterWaveDampingGas(pmb, cycle_dt, il, iu, jl, ju, kl, ku, pmb->phydro->u);
    if (Theta_Gas_Damping_Flag) {
      UpperWaveDampingGas(pmb, cycle_dt, il, iu, jl, ju, kl, ku, pmb->phydro->u);
      LowerWaveDampingGas(pmb, cycle_dt, il, iu, jl, ju, kl, ku, pmb->phydro->u);
    }
    if (NDUSTFLUIDS > 0) {
      if (Inner_Dust_Damping_Flag)
        InnerWaveDampingDust(pmb, cycle_dt, il, iu, jl, ju, kl, ku, pmb->pdustfluids->df_u);
      if (Outer_Dust_Damping_Flag)
        OuterWaveDampingDust(pmb, cycle_dt, il, iu, jl, ju, kl, ku, pmb->pdustfluids->df_u);
    }
  }
  PinDustBeforeRelease(pmb);  // after the gas damping, so dust follows the final gas state
}


void DustDensityPercentFloor(MeshBlock *pmb, int il, int iu, int jl, int ju,
            int kl, int ku, AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {

  int nc1 = pmb->ncells1;
  Real bumping_factor;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    //Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      //Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        //Real x1 = pmb->pcoord->x1v(i);
        GetCylCoordInSpherical(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
      }

      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real &dust_rho  = prim_df(rho_id, k, j, i);
          Real &dust_vel1 = prim_df(v1_id,  k, j, i);
          Real &dust_vel2 = prim_df(v2_id,  k, j, i);
          Real &dust_vel3 = prim_df(v3_id,  k, j, i);

          Real &dust_dens = cons_df(rho_id, k, j, i);
          Real &dust_mom1 = cons_df(v1_id,  k, j, i);
          Real &dust_mom2 = cons_df(v2_id,  k, j, i);
          Real &dust_mom3 = cons_df(v3_id,  k, j, i);

          Real density_init = DenProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i), initial_D2G[dust_id], Hratio[dust_id]);

          if (dust_rho < dust_percent_floor * density_init) {
            //std::cout << "P[" << Globals::my_rank << "]: dust_rho \%floor-ed at (r,phi,z)=(" << rad_arr(i) << ", " << phi_arr(i) << ", " << z_arr(i) << ")" << std::endl;
            dust_rho = 1.7782794100389227 * dust_percent_floor * density_init; // make it 0.25dex more
            bumping_factor = dust_dens / dust_rho;
            dust_vel1 *= bumping_factor; // also reduce velocity
            dust_vel2 *= bumping_factor;
            dust_vel3 *= bumping_factor;
            dust_dens = dust_rho;
          }
        }
      }
    }
	}
  return;
}


void MeshBlock::UserWorkInLoop() {

  Real mygam = peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);
  int dk     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  int kl = ks - dk;
  int ku = ke + dk;
  int jl = js - NGHOST;
  int ju = je + NGHOST;
  int il = is - NGHOST;
  int iu = ie + NGHOST;

  // Wave damping is done in PostUpdate, on active cells before the boundary exchange.

  // These optional legacy corrections still run after ghost filling. Their
  // restart reproducibility when they alter cells is outside the release fix.
  // If enabling either correction below, rerun bitwise restart tests with it
  // actually modifying cells, including checkpoints around dust release.
  if (Isothermal_Flag)
    LocalIsothermalEOS(this, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
  
  // Pre-release dust pinning is done in PostUpdate, before ghost filling.
   
  if ((NDUSTFLUIDS > 0) && (dust_percent_floor > 0.0))
    DustDensityPercentFloor(this, il, iu, jl, ju, kl, ku,
                            pdustfluids->df_w, pdustfluids->df_u);

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoop() {
  // RL: note that each node may have different number of meshblocks; thus, analyses
  // that involving all meshblocks and MPI_Reduce will need to be done here. 

  // Dust evolution uses the cycle-start release predicate, not a duty-cycle switch.

  return;
}
