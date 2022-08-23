using Test
using Distributed

function run_parallel( steps, Δt)

    σ, μ = 0.17, 0.0
    kx, α = 1.22, 0.02
    
    xmin, xmax = 0, 4*pi/kx
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
                                       efield_dofs,
                                       afield_dofs,
                                       domain);
    
    thdiag = TimeHistoryDiagnostics( maxwell_solver, 
                            kernel_smoother0, kernel_smoother1 );
    
    write_step!(thdiag, 0.0, spline_degree,
                        efield_dofs,  afield_dofs,
                        efield_poisson, 
                        propagator, particle_group)
    
    for j = 1:steps # loop over time
    
        operatorHE(propagator, particle_group, 0.5Δt)
        operatorHp(propagator, particle_group, 0.5Δt)
        operatorHA(propagator, particle_group, 0.5Δt)
        operatorHs(propagator, particle_group, 1.0Δt)
        operatorHA(propagator, particle_group, 0.5Δt)
        operatorHp(propagator, particle_group, 0.5Δt)
        operatorHE(propagator, particle_group, 0.5Δt)

    
        write_step!(thdiag, j * Δt, spline_degree, 
                        efield_dofs,  afield_dofs,
                        efield_poisson, 
                        propagator, particle_group)
    end

    return thdiag

end

@show nworkers()
if nworkers == 1
    addprocs(2)
end
@show nworkers()

steps, Δt = 100, 0.05

thdiag = run_simulation(steps, Δt)

ref = CSV.read("frame.csv", DataFrame)

@test thdiag.data[!, :KineticEnergy][1:steps] ≈ ref[!, :KineticEnergy][1:steps]
