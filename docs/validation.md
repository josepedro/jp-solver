# Validation

A case is not a demo. Each has a reference, a stated criterion and a tolerance fixed before the run.
A case that fails is reported as a failure, never renormalised until it passes.

The 3D Taylor-Green vortex is the canonical transition-to-turbulence benchmark: a smooth initial
field that stretches into vortex tubes and breaks down. The signature is the enstrophy history,
which rises, peaks, then falls, while the kinetic energy decays monotonically.

| Case | Reference | Criterion | Result |
|---|---|---|---|
| Uniform TGV, N=192, Re=1600 | Brachet et al. (1983); van Rees et al. (2011) | enstrophy peak time in [7.5, 10.0] (DNS 9.0), energy strictly decreasing, mass drift < 1e-4 | peak at t\*=8.93, energy monotonic, mass drift < 1e-4 |
| Static gold, refine all levels | the uniform TGV above | peak time within 0.3 in t\*, energy curves within 2 percent | peak time delta 0.108, energy within 1.98 percent |
| Dynamic gold, sensor-driven | the uniform TGV above | breakdown present, energy monotonic, mass held | passes |

The static gold test is the key correctness check for the adaptive path: when every level is refined
everywhere, the run must reproduce the uniform grid at the finest resolution. The viscosity is
consistent across levels by construction, since tau doubles the non-equilibrium scaling at each
level, so the finest-level lattice viscosity equals the uniform run at 4N.

An honest limit: partial refinement over-dissipates the small scales, so the sensor-driven run peaks
earlier and lower than the fully resolved DNS. The adaptive run is a demonstration of the method;
only the uniform solver matches the literature spectrum. This is stated so the visualisation is not
mistaken for a DNS-grade result.
