using Plots
using ProgressMeter
using Random
using SpinGEMPIC

function run_simulation( steps, Δt)

    σ, μ = sqrt(3.0/511), 0.0
    kx, α = 1/sqrt(2), 0.001
    xmin, xmax = 0, 2π/kx
    domain = [xmin, xmax, xmax - xmin]
    ∆t = 0.02
    nx = 128 
    n_particles = 200000
    mesh = Mesh( xmin, xmax, nx)
    spline_degree = 3
    
    df = CosSumGaussian{1,1,3}([[kx]], [α], [[σ]], [[μ]] )

    rng = MersenneTwister(42)
    
    mass, charge = 1.0, 1.0
    particle_group = ParticleGroup{1,1,3}( n_particles, mass, charge, 1)   
    sampler = ParticleSampler{1,1,3}(n_particles)
    
    sample!(rng, particle_group, sampler, df, mesh)
    
    kernel_smoother2 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree-2, :galerkin)    
    kernel_smoother1 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree-1, :galerkin)    
    kernel_smoother0 = ParticleMeshCoupling( domain, [nx], n_particles, spline_degree, :galerkin)

    rho = zeros(nx)

    maxwell_solver = Maxwell1DFEM(OneDGrid( xmin, xmax, nx), spline_degree)

    efield_poisson = zeros(Float64, nx)
    solve_poisson!( efield_poisson, particle_group, kernel_smoother0, maxwell_solver, rho )
    
    alpha1_sin_k(x) = -sqrt(3)*sin(2π * x / domain[3])
    alpha2_sin_k(x) = sqrt(3)*sin(2π * x / domain[3])
    beta_cos_k(x) = sqrt(3) * cos(2π * x / domain[3]) 
    
    efield_dofs = [efield_poisson, zeros(nx), zeros(nx)]
    afield_dofs = [zeros(nx), zeros(nx)]
    
    l2projection!( efield_dofs[2], maxwell_solver, beta_cos_k, spline_degree)
    l2projection!( efield_dofs[3], maxwell_solver, alpha2_sin_k, spline_degree)
    l2projection!( afield_dofs[1], maxwell_solver, alpha1_sin_k, spline_degree)
    l2projection!( afield_dofs[2], maxwell_solver, beta_cos_k, spline_degree)
        
    propagator = HamiltonianSplitting( maxwell_solver,
                                       kernel_smoother0, 
                                       kernel_smoother1, 
                                       kernel_smoother2, 
                                       particle_group,
                                       efield_dofs,
                                       afield_dofs,
                                       domain);
    
    efield_dofs_n = propagator.e_dofs
    bar = Progress(steos, 1) 
    
    anim = @animate for j = 1:steps # loop over time
    
        strang_splitting!(propagator, Δt, 1)
    
        xp = vcat([get_x(particle_group, i) for i in 1:n_particles]...)
        vp = vcat([get_v(particle_group, i) for i in 1:n_particles]'...)
        histogram2d(xp, vp, ylims=(-pi/kx,pi/kx),nbins=100)
        next!(bar)

    end  when mod1(j, 10) == 1

    return anim


end


steps, Δt = 100, 0.02

@time anim = run_simulation(steps, Δt)

gif(anim, "example2.gif", fps = 15)
