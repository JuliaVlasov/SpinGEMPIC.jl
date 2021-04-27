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

# -

# # Strong Landau Damping
#
# Electrostatic example of strong Landau damping
# $$
# f(x,v) =\frac{1}{2\pi\sigma^2}  \exp 
# \Big( - \frac{v_1^2 + v_2^2}{2\sigma^2} \Big)
# ( 1+\alpha \cos(k x)􏰁),
# $$
#
# The physical parameters 

# +
#σ, μ = sqrt(3.0/511), 0.0
σ, μ = 0.02, 0.0
##σ, μ = 0.02, 0.0
#σ, μ = 0.5, 1.5
##kx, α = 1.004355, 0.001
#kx, α = 1.2, 0.02
kx, α = 1.0706159423376036, 0.02 


xmin, xmax = 0, 2*2π/kx
domain = [xmin, xmax, xmax - xmin]
nx = 256
n_particles = 50000
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

xp = Vector{Float64}[] # particles data
for i in 1:n_particles
    push!(xp, vcat(get_x(particle_group,i), 
            get_v(particle_group,i),
            get_weights(particle_group,i)))
end
# -

xp = vcat([get_x(particle_group, i) for i in 1:n_particles]...)
vp = vcat([get_v(particle_group, i) for i in 1:n_particles]'...)
wp = vcat([get_weights(particle_group, i) for i in 1:n_particles]'...)
p = plot(layout=(2,1))
histogram!(p[1,1], xp, weights=wp, normalize= true, bins = 100, lab = "")
plot!(p[1,1], x-> (1+α*cos(kx*x))/(2π/kx), 0., 2π/kx, lab="")
histogram!(p[2,1], vp, weights=wp, normalize=true, bins = 100, lab = "")
plot!(p[2,1], v-> 1/sqrt(2*pi)/σ*(exp( - (v-μ)^2 / 2/σ/σ)), -6, 6, lab="")
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

# Init!ialize the field solver
maxwell_solver = Maxwell1DFEM(domain, nx, spline_degree)
# efield by Poisson
solve_poisson!( efield_poisson, particle_group, kernel_smoother0, maxwell_solver, rho )
sval = eval_uniform_periodic_spline_curve(spline_degree-1, efield_poisson)
plot( xg, sval )       

# +
# # +
# Initialize the arrays for the spline coefficients of the fields
##k0 = 12.0523
##E0 = 5.0 #10.0
##ww = 12.104827940833333
#k0=2.4
#k0=2.3311360403760835
#k0=12.052264124808531
#k0=5.12093250320881
k0=3.211847827012811
ww=sqrt(1.0+k0*k0)
E0=1.0
Ey(x) = E0*cos(k0*x)
Ez(x) = E0*sin(k0*x)
Ay(x) = -E0/ww*sin(k0*x)
Az(x) = E0/ww*cos(k0*x)

#e1(x) = α/kx*sin(2π * x / domain[3])
 

efield_dofs = [efield_poisson, zeros(Float64, nx), zeros(Float64, nx)]
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

#propagator = HamiltonianSplittingBoris( maxwell_solver,
#         kernel_smoother0, kernel_smoother1, particle_group,
#         efield_dofs, bfield_dofs, domain)



efield_dofs_n = propagator.e_dofs

thdiag = TimeHistoryDiagnostics( particle_group, maxwell_solver, 
                        kernel_smoother0, kernel_smoother1 );

steps, Δt = 15000, 0.02
mode1 = zeros(ComplexF64,steps)
mode2 = zeros(ComplexF64,steps)
store = zeros(ComplexF64,nx)

#=
test = zeros(Float64,nx)
nengliang = zeros(Float64,steps)
xp = vcat([get_x(particle_group, i) for i in 1:n_particles]...)
vp = vcat([get_v(particle_group, i) for i in 1:n_particles]'...)
histogram2d(xp, vp, ylims=(-pi/kx,pi/kx),nbins=200)
=#

s2_1 = zeros(Float64, n_particles)
s2_2 = zeros(Float64, n_particles)
s2_3 = zeros(Float64, n_particles)

s4_1 = zeros(Float64, n_particles)
s4_2 = zeros(Float64, n_particles)
s4_3 = zeros(Float64, n_particles)

s6_1 = zeros(Float64, n_particles)
s6_2 = zeros(Float64, n_particles)
s6_3 = zeros(Float64, n_particles)

s8_1 = zeros(Float64, n_particles)
s8_2 = zeros(Float64, n_particles)
s8_3 = zeros(Float64, n_particles)

s10_1 = zeros(Float64, n_particles)
s10_2 = zeros(Float64, n_particles)
s10_3 = zeros(Float64, n_particles)

s12_1 = zeros(Float64, n_particles)
s12_2 = zeros(Float64, n_particles)
s12_3 = zeros(Float64, n_particles)

s14_1 = zeros(Float64, n_particles)
s14_2 = zeros(Float64, n_particles)
s14_3 = zeros(Float64, n_particles)

electric = zeros(ComplexF64, steps, nx)

@showprogress 1 for j = 1:steps # loop over time

    # Strang splitting
    strang_splitting!(propagator, Δt, 1)
    # Lie splitting
    #lie_splitting!(propagator, Δt, 1)
    # Diagnostics

    solve_poisson!( efield_poisson, particle_group, 
                    kernel_smoother0, maxwell_solver, rho)
    
    write_step!(thdiag, j * Δt, spline_degree, 
                    efield_dofs,  afield_dofs,
                    efield_dofs_n, efield_poisson, propagator)
    test = zeros(Float64,nx)
    for i = 1:nx
        xi = 2*pi/kx/nx*(i-1)
        test[i] = evaluate(propagator.kernel_smoother_1, xi, propagator.e_dofs[1])
    end
    store .= fft(test)
    electric[j,:] .= store
        if j == 1
	global s0_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)    
	global s0_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)    
	global s0_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
	end
	
	if j == 200
	global s2_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)    
	global s2_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)    
	global s2_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
	end
   	
	if j == 400  
        global s4_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
        global s4_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
        global s4_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
        end

	if j == 600  
        global s6_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
        global s6_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
        global s6_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
        end

	if j == 800  
        global s8_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
        global s8_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
        global s8_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
        end

	if j == 1000  
        global s10_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
        global s10_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
        global s10_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
        end

	if j == 1200  
        global s12_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
        global s12_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
        global s12_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
        end

	if j == 2000  
        global s20_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
        global s20_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
        global s20_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
        end

	if j == 3000  
        global s30_1 = vcat([get_s1(particle_group, i) for i in 1:n_particles]...)
        global s30_2 = vcat([get_s2(particle_group, i) for i in 1:n_particles]...)
        global s30_3 = vcat([get_s3(particle_group, i) for i in 1:n_particles]...)
        end 
end


CSV.write("frame_main31.csv",thdiag.data)
#save("electric_main31.jld2", "electric", electric)

# -


# plot(thdiag.data[!,:Time], (thdiag.data[!,:PotentialEnergyE1] .+thdiag.data[!,:PotentialEnergyE2].+thdiag.data[!,:PotentialEnergyE3].+thdiag.data[!,:PotentialEnergyB2].+thdiag.data[!,:PotentialEnergyB3].+thdiag.data[!,:KineticEnergy] .+ thdiag.data[!,:Kineticspin]))

# plot(thdiag.data[!,:Time],thdiag.data[!,:Kineticspin])

# +
plot(thdiag.data[!,:Time], log10.(abs.(mode1)))
using Plots


# -

#=
test = zeros(Float64,nx)
for i = 1:nx
    xi = 2*pi/kx/nx*(i-1)
    test[i] = evaluate(propagator.kernel_smoother_1, xi, propagator.a_dofs[1])
end
plot!(test)
=#
#=
xp = vcat([get_x(particle_group, i) for i in 1:n_particles]...)
vp = vcat([get_v(particle_group, i) for i in 1:n_particles]'...)
histogram2d(xp, vp, ylims=(-6,6),nbins=100,xlabel="x", ylabel = "p")
=#
