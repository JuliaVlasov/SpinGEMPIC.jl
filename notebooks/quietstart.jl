using Plots
using Random
using SpinGEMPIC

import GEMPIC: OneDGrid

σ, μ = 0.17, 0.0
kx, α = 1.22, 0.02

xmin, xmax = 0, 4pi/kx
nx = 128
mesh = OneDGrid( xmin, xmax, nx)

n_particles = 20000

df = CosGaussian(kx, α, σ, μ)

rng = MersenneTwister(123)
mass, charge = 1.0, 1.0

particle_group = ParticleGroup( n_particles, mass, charge, 1)   
sample!(rng, particle_group, df, mesh, method = :quietstart)


xp = view(particle_group.array, 1, :)
vp = view(particle_group.array, 2, :)
s1 = view(particle_group.array, 3, :)
s2 = view(particle_group.array, 4, :)
s3 = view(particle_group.array, 5, :)
wp = view(particle_group.array, 6, :)

p = plot(layout=(6,1))
#sphereplot!(p[1], particle_group, 10) # plot 2000 particles
histogram!(p[2], s1, weights=wp, normalize=true, bins = 100, lab = "")
histogram!(p[3], s2, weights=wp, normalize=true, bins = 100, lab = "")
histogram!(p[4], s3, weights=wp, normalize=true, bins = 100, lab = "")
plot!(p[4], x -> (1 + x / 2) / 2, -1, 1, lab="")
histogram!(p[5], xp, weights=wp, normalize= true, bins = 100, lab = "")
plot!(p[5], x-> (1+α*cos(kx*x))/(4π/kx), 0., 4π/kx, lab="")
ylims!(p[5], (0.09,0.11))
histogram!(p[6], vp, weights=wp, normalize=true, bins = 100, lab = "")
plot!(p[6], v-> 1/sqrt(2pi)/σ*(exp(-(v-μ)^2 / 2/σ/σ)), -1, 1, lab="")
