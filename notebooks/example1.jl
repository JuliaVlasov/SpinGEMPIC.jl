using DataFrames
using CSV
using Plots
using ProgressMeter
using Random
using SpinGEMPIC
using TimerOutputs

import SpinGEMPIC: set_common_weight
import SpinGEMPIC: get_s1, get_s2, get_s3
import SpinGEMPIC: set_s1, set_s2, set_s3
import SpinGEMPIC: set_weights, get_weights
import SpinGEMPIC: set_x, set_v

import SpinGEMPIC: operatorHE
import SpinGEMPIC: operatorHp
import SpinGEMPIC: operatorHA
import SpinGEMPIC: operatorHs

import GEMPIC: OneDGrid, Maxwell1DFEM
import GEMPIC: l2projection!

const to = TimerOutput()

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
    mesh = OneDGrid( xmin, xmax, nx)
    spline_degree = 3
    
    df = CosSumGaussian{1,1,3}([[kx]], [α], [[σ]], [[μ]] )
    
    rng = MersenneTwister(123)
    mass, charge = 1.0, 1.0
    
    particle_group = ParticleGroup{1,1,3}( n_particles, mass, charge, 1)   
    sampler = ParticleSampler{1,1,3}( n_particles)
    sample!(rng, particle_group, sampler, df, mesh)
    set_common_weight(particle_group, (1.0/n_particles))

    kernel_smoother2 = ParticleMeshCoupling( mesh, n_particles, spline_degree-2, :galerkin) 
    kernel_smoother1 = ParticleMeshCoupling( mesh, n_particles, spline_degree-1, :galerkin)    
    kernel_smoother0 = ParticleMeshCoupling( mesh, n_particles, spline_degree, :galerkin)
    
    maxwell_solver = Maxwell1DFEM(mesh, spline_degree)

    rho = zeros(Float64, nx)
    efield_poisson = zeros(Float64, nx)
    
    solve_poisson!( efield_poisson, particle_group, kernel_smoother0, maxwell_solver, rho )
    
    k0 = 2*kx 
    E0 = 0.325 
    ww = 2.63
    Ey(x) = E0*cos(k0*x)
    Ez(x) = E0*sin(k0*x)
    Ay(x) = -E0/ww*sin(k0*x)
    Az(x) = E0/ww*cos(k0*x)
      
    efield_dofs = [ zeros(nx), zeros(nx), zeros(nx)]
    efield_dofs[1] .= efield_poisson 
    afield_dofs = [zeros(nx), zeros(nx)]
    
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
    
        @timeit to "HE" operatorHE(propagator, 0.5Δt)
        @timeit to "Hp" operatorHp(propagator, 0.5Δt)
        @timeit to "HA" operatorHA(propagator, 0.5Δt)
        @timeit to "Hs" operatorHs(propagator, 1.0Δt)
        @timeit to "HA" operatorHA(propagator, 0.5Δt)
        @timeit to "Hp" operatorHp(propagator, 0.5Δt)
        @timeit to "HE" operatorHE(propagator, 0.5Δt)

    
        write_step!(thdiag, j * Δt, spline_degree, 
                        efield_dofs,  afield_dofs,
                        efield_dofs_n, efield_poisson, propagator)
    end

    return thdiag

end

steps, Δt = 100, 0.05

thdiag = run_simulation(steps, Δt)

show(to)

ref = CSV.read("frame.csv", DataFrame)

time = thdiag.data[!, :Time]
kenergy = thdiag.data[!, :KineticEnergy]
plot( time, kenergy, xlabel = "time", ylabel = "kinetic energy", label = "new")

time = ref[!, :Time]
kenergy = ref[!, :KineticEnergy]
plot!( time, kenergy, xlabel = "time", ylabel = "kinetic energy", label = "old")

