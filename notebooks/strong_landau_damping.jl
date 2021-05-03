# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Julia 1.3.0
#     language: julia
#     name: julia-1.3
# ---

# +
using ProgressMeter, Plots
using CSV
using JLD2
using JLD
using FileIO

include("../src/mesh.jl")
include("../src/distributions.jl")
include("../src/low_level_bsplines.jl")
include("../src/splinepp.jl")
include("../src/maxwell_1d_fem.jl")
include("../src/particle_group.jl")
include("../src/particle_mesh_coupling.jl")
include("../src/hamiltonian_splitting.jl")
include("../src/hamiltonian_splitting_boris.jl")
include("../src/particle_sampling.jl")
include("../src/diagnostics.jl")

# Initial condition of the form 
# $$
# f(x,v) =\frac{1}{2\pi\sigma^2}  exp(-0.5 (v-μ)^2 / \sigma^2 ) * ( 1+\alpha \cos(kx x)􏰁),
# $$


# The physical parameters 

# for Laser plasma test (without spin)
#σ, μ = sqrt(3.0/511), 0.0
#kx, α = 1/sqrt(2), 0.001

# for test with spin
σ, μ = 0.17, 0.0
kx, α = 1.22, 0.02

xmin, xmax = 0, 4*pi/kx
domain = [xmin, xmax, xmax - xmin]
nx = 128
n_particles = 20000
mesh = Mesh( xmin, xmax, nx)
spline_degree = 3

df = CosSumGaussian{1,1,3}([[kx]], [α], [[σ]], [[μ]] )


mass, charge = 1.0, 1.0

particle_group2 = ParticleGroup{1,1,3}( n_particles, mass, charge, 1)   
#sampler = ParticleSampler{1,1,3}( :sobol, true, n_particles)
sampler = ParticleSampler{1,1,3}( :sobol, false, n_particles)
sample!(particle_group2, sampler, df, mesh)

particle_group = ParticleGroup{1,1,3}( n_particles, mass, charge, 1)   
set_common_weight(particle_group, (1.0/n_particles))
for  i_part = 1:n_particles
    x = zeros( 1 )
    v = zeros( 1 )
    s = zeros( 3 )
    w = zeros( 1 )
    x = get_x(particle_group2, i_part)
    v = get_v(particle_group2, i_part)
    s[1] =  get_s1(particle_group2, i_part)
    s[2] =  get_s2(particle_group2, i_part)
    s[3] =  get_s3(particle_group2, i_part)
    w = get_weights(particle_group2, i_part)
    set_x(particle_group, i_part, x[1])
    set_v(particle_group, i_part, v[1])
    set_s1(particle_group, i_part, s[1])
    set_s2(particle_group, i_part, s[2])
    set_s3(particle_group, i_part, s[3])
    set_weights(particle_group, i_part, w[1])
end

xp = Vector{Float64}[] 
for i in 1:n_particles
    push!(xp, vcat(get_x(particle_group,i), 
            get_v(particle_group,i),
            get_weights(particle_group,i)))
end

xp = vcat([get_x(particle_group, i) for i in 1:n_particles]...)
vp = vcat([get_v(particle_group, i) for i in 1:n_particles]'...)
wp = vcat([get_weights(particle_group, i) for i in 1:n_particles]'...)
p = plot(layout=(2,1))

# Some initial diagnostics 
#histogram!(p[1,1], xp, weights=wp, normalize= true, bins = 100, lab = "")
#plot!(p[1,1], x-> (1+α*cos(kx*x))/(2π/kx), 0., 2π/kx, lab="")
#histogram!(p[2,1], vp, weights=wp, normalize=true, bins = 100, lab = "")
#plot!(p[2,1], v-> 1/sqrt(2*pi)/σ*(exp( - (v-μ)^2 / 2/σ/σ)), -6, 6, lab="")
#plot!(p[2,1], v-> 1/2/sqrt(2*pi)/σ*(exp( - (v-μ)^2 / 2/σ/σ)+exp( - (v+μ)^2 / 2/σ/σ)), -6, 6, lab="")


kernel_smoother2 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree-2, :galerkin) 
kernel_smoother1 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree-1, :galerkin)    
kernel_smoother0 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree, :galerkin)

rho = zeros(Float64, nx)
ex = zeros(Float64, nx)
maxwell_solver = Maxwell1DFEM(domain, nx, spline_degree)
solve_poisson!( ex, particle_group, 
                kernel_smoother0, maxwell_solver, rho)
xg = LinRange(xmin, xmax, nx)
sval = eval_uniform_periodic_spline_curve(spline_degree-1, rho)
plot( xg, sval )

efield_poisson = zeros(Float64, nx)

# Initialize the field solver
maxwell_solver = Maxwell1DFEM(domain, nx, spline_degree)
# efield by Poisson
solve_poisson!( efield_poisson, particle_group, kernel_smoother0, maxwell_solver, rho )
sval = eval_uniform_periodic_spline_curve(spline_degree-1, efield_poisson)
# Initial diagnostics
#plot( xg, sval )
#e1(x) = α/kx*sin(2π * x / domain[3])


# Initialize the arrays for the spline coefficients of the fields with a circularly polarized wave
k0 = 2*kx 
E0 = 0.325 
ww = 2.63
Ey(x) = E0*cos(k0*x)
Ez(x) = E0*sin(k0*x)
Ay(x) = -E0/ww*sin(k0*x)
Az(x) = E0/ww*cos(k0*x)
  
#efield_dofs = [efield_poisson, zeros(Float64, nx), zeros(Float64, nx)]
e_dofs = [ zeros(nx), zeros(nx), zeros(nx)]
e_dofs[1] .= efield_poisson 
afield_dofs = [zeros(Float64, nx), zeros(Float64, nx)]
#l2projection!( efield_poisson, maxwell_solver, e1, spline_degree-1)
l2projection!( efield_dofs[2], maxwell_solver, Ey, spline_degree)
l2projection!( efield_dofs[3], maxwell_solver, Ez, spline_degree)
l2projection!( afield_dofs[1], maxwell_solver, Ay, spline_degree)
l2projection!( afield_dofs[2], maxwell_solver, Az, spline_degree)
    
propagator = HamiltonianSplitting( maxwell_solver,
                                   kernel_smoother0, 
                                   kernel_smoother1, 
                                   kernel_smoother2,
                                   particle_group,
                                   efield_dofs,
                                   afield_dofs,
                                   domain);

efield_dofs_n = propagator.e_dofs

# Time parameters 
thdiag = TimeHistoryDiagnostics( particle_group, maxwell_solver, 
                        kernel_smoother0, kernel_smoother1 );

steps, Δt = 40000, 0.1
mode1 = zeros(ComplexF64,steps)
mode2 = zeros(ComplexF64,steps)
store = zeros(ComplexF64,nx)

# Diagnostics parameters 
test = zeros(Float64,nx)
nengliang = zeros(Float64,steps)
xp = vcat([get_x(particle_group, i) for i in 1:n_particles]...)
vp = vcat([get_v(particle_group, i) for i in 1:n_particles]'...)
#histogram2d(xp, vp, ylims=(-pi/kx,pi/kx),nbins=200)

s1_1 = zeros(Float64, n_particles)
s1_2 = zeros(Float64, n_particles)
s1_3 = zeros(Float64, n_particles)

s2_1 = zeros(Float64, n_particles)
s2_2 = zeros(Float64, n_particles)
s2_3 = zeros(Float64, n_particles)

s3_1 = zeros(Float64, n_particles)
s3_2 = zeros(Float64, n_particles)
s3_3 = zeros(Float64, n_particles)

s4_1 = zeros(Float64, n_particles)
s4_2 = zeros(Float64, n_particles)
s4_3 = zeros(Float64, n_particles)

s5_1 = zeros(Float64, n_particles)
s5_2 = zeros(Float64, n_particles)
s5_3 = zeros(Float64, n_particles)

s6_1 = zeros(Float64, n_particles)
s6_2 = zeros(Float64, n_particles)
s6_3 = zeros(Float64, n_particles)

s7_1 = zeros(Float64, n_particles)
s7_2 = zeros(Float64, n_particles)
s7_3 = zeros(Float64, n_particles)

s8_1 = zeros(Float64, n_particles)
s8_2 = zeros(Float64, n_particles)
s8_3 = zeros(Float64, n_particles)


s9_1 = zeros(Float64, n_particles)
s9_2 = zeros(Float64, n_particles)
s9_3 = zeros(Float64, n_particles)

write_step!(thdiag, 0.0, spline_degree,
                    efield_dofs,  afield_dofs,
                    efield_dofs_n, efield_poisson, propagator)

global s1_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
global s1_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
global s1_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)

@showprogress 1 for j = 1:steps # loop over time

    # Time splitting: Strang or Lie 	      	      
    strang_splitting!(propagator, Δt, 1)
    #lie_splitting!(propagator, Δt, 1)

    # Diagnostics
    write_step!(thdiag, j * Δt, spline_degree, 
                    efield_dofs,  afield_dofs,
                    efield_dofs_n, efield_poisson, propagator)

    if j == 2500
	global s2_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
	global s2_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
	global s2_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
    end
   	
    if j == 5000  
        global s3_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
	global s3_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
	global s3_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
    end

    if j == 7500  
        global s4_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
	global s4_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
	global s4_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
    end

    if j == 10000  
        global s5_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
	global s5_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
	global s5_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
    end

    if j == 12500  
        global s6_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
	global s6_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
	global s6_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
   end

   if j == 15000  
        global s7_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
	global s7_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
	global s7_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
   end

   if j == 25000  
        global s8_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
	global s8_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
	global s8_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
   end
	
   if j == 37500
        global s9_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
	global s9_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
	global s9_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
   end

end


CSV.write("frame.csv",thdiag.data)

save("s1_1.jld2", "s1_1", s1_1)
save("s1_2.jld2", "s1_2", s1_2)
save("s1_3.jld2", "s1_3", s1_3)
save("s2_1.jld2", "s2_1", s2_1)
save("s2_2.jld2", "s2_2", s2_2)
save("s2_3.jld2", "s2_3", s2_3)
save("s3_1.jld2", "s3_1", s3_1)
save("s3_2.jld2", "s3_2", s3_2)
save("s3_3.jld2", "s3_3", s3_3)
save("s4_1.jld2", "s4_1", s4_1)
save("s4_2.jld2", "s4_2", s4_2)
save("s4_3.jld2", "s4_3", s4_3)
save("s5_1.jld2", "s5_1", s5_1)
save("s5_2.jld2", "s5_2", s5_2)
save("s5_3.jld2", "s5_3", s5_3)
save("s6_1.jld2", "s6_1", s6_1)
save("s6_2.jld2", "s6_2", s6_2)
save("s6_3.jld2", "s6_3", s6_3)
save("s7_1.jld2", "s7_1", s7_1)
save("s7_2.jld2", "s7_2", s7_2)
save("s7_3.jld2", "s7_3", s7_3)
save("s8_1.jld2", "s8_1", s8_1)
save("s8_2.jld2", "s8_2", s8_2)
save("s8_3.jld2", "s8_3", s8_3)
save("s9_1.jld2", "s9_1", s9_1)
save("s9_2.jld2", "s9_2", s9_2)
save("s9_3.jld2", "s9_3", s9_3)


