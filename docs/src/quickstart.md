# Quickstart


## Import packages 

```@example quickstart
using Plots
using Random
using SpinGEMPIC

import SpinGEMPIC: set_common_weight

import GEMPIC: OneDGrid, Maxwell1DFEM
import GEMPIC:  l2projection!, eval_uniform_periodic_spline_curve
```

## Physics parameters

```@example quickstart
σ, μ = 0.17, 0.0
kx, α = 1.22, 0.02
```

## Initialize mesh

```@example quickstart
xmin, xmax = 0, 4pi/kx
nx = 128
mesh = OneDGrid( xmin, xmax, nx)
```

## Initialize particles


```@example quickstart
n_particles = 10000

df = CosGaussian(kx, α, σ, μ)

rng = MersenneTwister(123)
mass, charge = 1.0, 1.0

particle_group = ParticleGroup( n_particles, mass, charge, 1)   
sample!(rng, particle_group, df, mesh)
set_common_weight(particle_group, (1.0/n_particles))
sphereplot(particle_group)
```

You can plot ten times less particles

```@example quickstart
sphereplot(particle_group, 10)
```


## Initialize Maxwell solver

```@example quickstart
spline_degree = 3

kernel_smoother0 = ParticleMeshCoupling( mesh, n_particles, spline_degree)

maxwell_solver = Maxwell1DFEM(mesh, spline_degree)

rho = zeros(nx)
efield_poisson = zeros(nx)

solve_poisson!( efield_poisson, particle_group, kernel_smoother0, maxwell_solver, rho )
sval = eval_uniform_periodic_spline_curve(spline_degree-1, efield_poisson)
plot(LinRange(xmin, xmax, nx), sval, label="Ex")
```
