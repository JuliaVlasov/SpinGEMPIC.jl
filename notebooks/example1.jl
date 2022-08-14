using DataFrames
using CSV
using Plots
using ProgressMeter
using Random
using SpinGEMPIC

import SpinGEMPIC: set_common_weight
import SpinGEMPIC: get_s1, get_s2, get_s3
import SpinGEMPIC: set_s1, set_s2, set_s3
import SpinGEMPIC: set_weights, get_weights
import SpinGEMPIC: set_x, set_v

import GEMPIC: OneDGrid, Maxwell1DFEM
import GEMPIC: eval_uniform_periodic_spline_curve
import GEMPIC: l2projection!

"""
Test corresponding to Fig. 4 of the JPP paper 
"""
function run_simulation( steps, Δt)

    σ, μ = 0.17, 0.0
    kx, α = 1.22, 0.02
    
    xmin, xmax = 0, 4*pi/kx
    domain = [xmin, xmax, xmax - xmin]
    nx = 128
    n_particles = 20000
    mesh = Mesh( xmin, xmax, nx)
    spline_degree = 3
    
    df = CosSumGaussian{1,1,3}([[kx]], [α], [[σ]], [[μ]] )
    
    rng = MersenneTwister(123)
    mass, charge = 1.0, 1.0
    
    particle_group = ParticleGroup{1,1,3}( n_particles, mass, charge, 1)   
    sampler = ParticleSampler{1,1,3}( n_particles)
    sample!(rng, particle_group, sampler, df, mesh)
    set_common_weight(particle_group, (1.0/n_particles))

    for  i_part = 1:n_particles
        x = zeros( 1 )
        v = zeros( 1 )
        s = zeros( 3 )
        w = zeros( 1 )
        x = get_x(particle_group, i_part)
        v = get_v(particle_group, i_part)
        s[1] =  get_s1(particle_group, i_part)
        s[2] =  get_s2(particle_group, i_part)
        s[3] =  get_s3(particle_group, i_part)
        w = get_weights(particle_group, i_part)
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
    
    kernel_smoother2 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree-2, :galerkin) 
    kernel_smoother1 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree-1, :galerkin)    
    kernel_smoother0 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree, :galerkin)
    
    rho = zeros(Float64, nx)
    ex = zeros(Float64, nx)

    maxwell_solver = Maxwell1DFEM(OneDGrid( xmin, xmax, nx), spline_degree)

    solve_poisson!( ex, particle_group, 
                    kernel_smoother0, maxwell_solver, rho)
    xg = LinRange(xmin, xmax, nx)
    sval = eval_uniform_periodic_spline_curve(spline_degree-1, rho)
    
    efield_poisson = zeros(Float64, nx)
    
    solve_poisson!( efield_poisson, particle_group, kernel_smoother0, maxwell_solver, rho )
    sval = eval_uniform_periodic_spline_curve(spline_degree-1, efield_poisson)
    
    
    k0 = 2*kx 
    E0 = 0.325 
    ww = 2.63
    Ey(x) = E0*cos(k0*x)
    Ez(x) = E0*sin(k0*x)
    Ay(x) = -E0/ww*sin(k0*x)
    Az(x) = E0/ww*cos(k0*x)
      
    efield_dofs = [ zeros(nx), zeros(nx), zeros(nx)]
    efield_dofs[1] .= efield_poisson 
    afield_dofs = [zeros(Float64, nx), zeros(Float64, nx)]
    
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
    
    thdiag = TimeHistoryDiagnostics( particle_group, maxwell_solver, 
                            kernel_smoother0, kernel_smoother1 );
    
    write_step!(thdiag, 0.0, spline_degree,
                        efield_dofs,  afield_dofs,
                        efield_dofs_n, efield_poisson, propagator)
    
    @showprogress 1 for j = 1:steps # loop over time
    
        strang_splitting!(propagator, Δt, 1)
    
        write_step!(thdiag, j * Δt, spline_degree, 
                        efield_dofs,  afield_dofs,
                        efield_dofs_n, efield_poisson, propagator)
    end

    return thdiag

end

steps, Δt = 300, 0.05

thdiag = run_simulation(steps, Δt)
ref = CSV.read("frame.csv", DataFrame)

time = thdiag.data[!, :Time]
kenergy = thdiag.data[!, :KineticEnergy]
plot( time, kenergy, xlabel = "time", ylabel = "kinetic energy", label = "new")

time = ref[!, :Time]
kenergy = ref[!, :KineticEnergy]
plot!( time, kenergy, xlabel = "time", ylabel = "kinetic energy", label = "old")

