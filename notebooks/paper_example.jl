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

import SpinGEMPIC: operatorHE
import SpinGEMPIC: operatorHp
import SpinGEMPIC: operatorHA
import SpinGEMPIC: operatorHs

import GEMPIC: OneDGrid, Maxwell1DFEM
import GEMPIC: l2projection!

import HDF5
import Printf

function save_spin(istep, pg)

    filename = Printf.@sprintf("spin-%05i", istep)
    HDF5.h5open(filename * ".h5", "w") do file
        HDF5.write(file, "s1", pg.array[3,:])  
        HDF5.write(file, "s2", pg.array[4,:]) 
        HDF5.write(file, "s3", pg.array[5,:])
    end

end

"""
Test corresponding to Fig. 4 of the JPP paper 
"""
function run_simulation( steps, Δt)

    σ, μ = 0.17, 0.0
    kx, α = 1.22, 0.02
    
    xmin, xmax = 0, 4pi/kx
    domain = [xmin, xmax, xmax - xmin]
    nx = 128
    n_particles = 20000
    mesh = OneDGrid( xmin, xmax, nx)
    spline_degree = 3
    
    df = CosGaussian(kx, α, σ, μ )
    
    rng = MersenneTwister(123)
    mass, charge = 1.0, 1.0
    
    particle_group = ParticleGroup( n_particles, mass, charge, 1)   
    sample!(rng, particle_group, df, mesh)
    set_common_weight(particle_group, (1.0/n_particles))

    kernel_smoother2 = ParticleMeshCoupling( mesh, n_particles, spline_degree-2) 
    kernel_smoother1 = ParticleMeshCoupling( mesh, n_particles, spline_degree-1)    
    kernel_smoother0 = ParticleMeshCoupling( mesh, n_particles, spline_degree)
    
    maxwell_solver = Maxwell1DFEM(mesh, spline_degree)

    rho = zeros(nx)
    efield_poisson = zeros(nx)
    
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
                                       efield_dofs,
                                       afield_dofs,
                                       domain);
    
    thdiag = TimeHistoryDiagnostics( maxwell_solver, 
                            kernel_smoother0, kernel_smoother1 )
    
    write_step!(thdiag, 0.0, spline_degree,
                        efield_dofs,  afield_dofs,
                        efield_poisson, 
                        propagator, particle_group)
    
    save_spin(1, particle_group)

    @showprogress 1 for j = 1:steps # loop over time
    
        strang_splitting!(propagator, particle_group, Δt, 1)
    
        write_step!(thdiag, j * Δt, spline_degree, 
                        efield_dofs,  afield_dofs,
                        efield_poisson, 
                        propagator, particle_group)

        if mod(j, 1000) == 0
            save_spin(j, particle_group)
        end

    end

    return thdiag

end

steps, Δt = 80000, 0.05

thdiag = run_simulation(steps, Δt)

CSV.write("data_paper.csv",thdiag.data)

ref = CSV.read("frame.csv", DataFrame)

time = thdiag.data[!, :Time]
kenergy = thdiag.data[!, :KineticEnergy]
plot( time, kenergy, xlabel = "time", ylabel = "kinetic energy", label = "new")

time = ref[!, :Time]
kenergy = ref[!, :KineticEnergy]
plot!( time, kenergy, xlabel = "time", ylabel = "kinetic energy", label = "old")

