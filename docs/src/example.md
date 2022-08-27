# Paper example

Test corresponding to Fig. 4 of the [paper](https://hal.inria.fr/hal-03148534v2/document) 

```@example paper
using Plots
using Random
using SpinGEMPIC

import GEMPIC: OneDGrid, Maxwell1DFEM
import GEMPIC: l2projection!

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
    sample!(rng, particle_group, df, mesh, method = :quietstart)

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
    
    for j = 1:steps # loop over time
    
        strang_splitting!(propagator, particle_group, Δt, 1)
    
        write_step!(thdiag, j * Δt, spline_degree, 
                        efield_dofs,  afield_dofs,
                        efield_poisson, 
                        propagator, particle_group)

    end

    return thdiag

end

steps, Δt = 100, 0.05

thdiag = run_simulation(steps, Δt)

time = thdiag.data[!, :Time]
kenergy = thdiag.data[!, :KineticEnergy]
plot( time, kenergy, xlabel = "time", ylabel = "kinetic energy", label = "new")
```

