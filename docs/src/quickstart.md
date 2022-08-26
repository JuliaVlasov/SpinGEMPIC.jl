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
n_particles = 20000

df = CosGaussian(kx, α, σ, μ)

rng = MersenneTwister(123)
mass, charge = 1.0, 1.0

particle_group = ParticleGroup( n_particles, mass, charge, 1)   
sample!(rng, particle_group, df, mesh, method = :quietstart)
```

```@example quickstart
sphereplot(particle_group, 10) # plot 2000 particles
```

```@example quickstart
xp = view(particle_group.array, 1, :)
vp = view(particle_group.array, 2, :)
s1 = view(particle_group.array, 3, :)
s2 = view(particle_group.array, 4, :)
s3 = view(particle_group.array, 5, :)
wp = view(particle_group.array, 6, :)

p = plot(layout=(3,1))
histogram!(p[1], s1, weights=wp, normalize=true, bins = 100, lab = "")
histogram!(p[2], s2, weights=wp, normalize=true, bins = 100, lab = "")
histogram!(p[3], s3, weights=wp, normalize=true, bins = 100, lab = "")
plot!(p[3], x -> (1 + x / 2) / 2, -1, 1, lab="")
```

```@example quickstart
p = plot(layout=(2,1))
histogram!(p[1], xp, weights=wp, normalize= true, bins = 100, lab = "")
plot!(p[1], x-> (1+α*cos(kx*x))/(4π/kx), 0., 4π/kx, lab="")
ylims!(p[1], (0.09,0.11))
histogram!(p[2], vp, weights=wp, normalize=true, bins = 100, lab = "")
plot!(p[2], v-> 1/sqrt(2pi)/σ*(exp(-(v-μ)^2 / 2/σ/σ)), -1, 1, lab="")
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
