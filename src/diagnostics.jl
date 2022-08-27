
using DataFrames

export solve_poisson!

"""
   solve_poisson!( efield, particle_group, kernel_smoother, maxwell_solver, rho )

Accumulate rho and solve Poisson
 - `particle_group` : Particles
 - `maxwell_solver` : Maxwell solver (FEM 1D)
 - `kernel_smoother_0` : Particle-Mesh method
 - `rho` : preallocated array for Charge density
 - `efield_dofs` : spline coefficients of electric field (1D)
"""
function solve_poisson!(
    efield_dofs::Vector{Float64},
    particle_group::ParticleGroup,
    kernel_smoother_0::ParticleMeshCoupling,
    maxwell_solver::Maxwell1DFEM,
    rho::Vector{Float64},
)

    fill!(rho, 0.0)
    for i_part = 1:particle_group.n_particles
        xi = get_x(particle_group, i_part)
        wi = get_charge(particle_group, i_part)
        add_charge!(rho, kernel_smoother_0, xi, wi)
    end
    compute_e_from_rho!(efield_dofs, maxwell_solver, rho)

end


"""
    pic_diagnostics_transfer( particle_group, kernel_smoother_0, 
                            kernel_smoother_1, efield_dofs, transfer)

Compute ``\\sum_{particles} w_p ( v_1,p e_1(x_p) + v_2,p e_2(x_p)) ``

- `particle_group`   
- `kernel_smoother_0`  : Kernel smoother (order p+1)
- `kernel_smoother_1`  : Kernel smoother (order p)   
- `efield_dofs` : coefficients of efield

"""
function pic_diagnostics_transfer(
    particle_group,
    kernel_smoother_0,
    kernel_smoother_1,
    efield_dofs,
)

    transfer = 0.0
    for i_part = 1:particle_group.n_particles

        xi = get_x(particle_group, i_part)
        wi = get_charge(particle_group, i_part)
        vi = get_v(particle_group, i_part)

        efield_1 = evaluate(kernel_smoother_1, xi, efield_dofs[1])
        efield_2 = evaluate(kernel_smoother_0, xi, efield_dofs[2])

        transfer += (vi * efield_1) * wi

    end

    transfer

end


export TimeHistoryDiagnostics

"""
    TimeHistoryDiagnostics( maxwell_solver, kernel_smoother_0, kernel_smoother_1 )

Context to save and plot diagnostics

- `maxwell_solver` : Maxwell solver
- `kernel_smoother_0` : Mesh coupling operator
- `kernel_smoother_1` : Mesh coupling operator
- `data` : DataFrame containing time history values

Outputs

- KineticEnergy: ``\\frac{1}{2} \\sum \\omega_i v_i^2``
- Kineticspi: Zeeman energy

Momentums: compute integrals of f 

- Momentum1 : ``\\sum x_i  \\omega_i``
- Momentum2 : ``\\sum x_i  \\omega_i  s1``
- Momentum3 : ``\\sum x_i  \\omega_i  s2``
- Momentum4 : ``\\sum x_i  \\omega_i  s3``
- Momentum5 : ``\\sum \\omega_i  s1``
- Momentum6 : ``\\sum \\omega_i  s2``
- Momentum7 : ``\\sum \\omega_i  s3``
- Momentum8 : `` A_y \\sum div(\\rho)  \\omega_i s2 + A_z div(\\rho) \\omega_i s3``
- Momentum9 :  ``- A_y div(\\rho) \\omega_i s1`` 
- Momentum10 : ``- A_z div(\\rho) \\omega_i s1``
- PotentialEnergyE1 : ``\\frac{1}{2} E_x^2``
- PotentialEnergyE2 : ``\\frac{1}{2} E_y^2``
- PotentialEnergyE3 : ``\\frac{1}{2} E_z^2``
- PotentialEnergyB2 : ``\\frac{1}{2} B_y^2``
- PotentialEnergyB3 : ``\\frac{1}{2} B_z^2``
- Transfer : ``\\sum (v_i \\cdot e_{x,i}) w_i ``
- ErrorPoisson : difference between ``E_x`` computed with Maxwell Solver et ``E_x`` computed from charge
"""
mutable struct TimeHistoryDiagnostics

    maxwell_solver::Maxwell1DFEM
    kernel_smoother_0::ParticleMeshCoupling
    kernel_smoother_1::ParticleMeshCoupling
    data::DataFrame

    function TimeHistoryDiagnostics(
        maxwell_solver::Maxwell1DFEM,
        kernel_smoother_0::ParticleMeshCoupling,
        kernel_smoother_1::ParticleMeshCoupling,
    )


        data = DataFrame(
            Time = Float64[],
            KineticEnergy = Float64[],
            Kineticspin = Float64[],
            Momentum1 = Float64[],
            Momentum2 = Float64[],
            Momentum3 = Float64[],
            Momentum4 = Float64[],
            Momentum5 = Float64[],
            Momentum6 = Float64[],
            Momentum7 = Float64[],
            Momentum8 = Float64[],
            Momentum9 = Float64[],
            Momentum10 = Float64[],
            PotentialEnergyE1 = Float64[],
            PotentialEnergyE2 = Float64[],
            PotentialEnergyE3 = Float64[],
            PotentialEnergyB2 = Float64[],
            PotentialEnergyB3 = Float64[],
            Transfer = Float64[],
            ErrorPoisson = Float64[]
        )


        new(maxwell_solver, kernel_smoother_0, kernel_smoother_1, data)
    end
end

export write_step!

"""
    write_step!( thdiag, time, degree, efield_dofs, bfield_dofs,
                 efield_poisson, propagator, particle_group)

write diagnostics for PIC
- `time` : Time
- `afield_dofs[1]` : Magnetic Potential Ay
- `afield_dofs[2]` : Magnetic Potential Az
- `efield_dofs[1]` : Longitudinal Electric field Ex
- `efield_poisson` : Electric field compute from Poisson equation
- `efield_dofs[2]` : Ey
- `efield_dofs[3]` : Ez
- `degree` : Spline degree
"""

function write_step!(
    thdiag::TimeHistoryDiagnostics,
    time,
    degree,
    efield_dofs,
    afield_dofs,
    efield_poisson,
    propagator,
    particle_group
)

    diagnostics = zeros(Float64, 12)
    potential_energy = zeros(Float64, 5)
    HH = 0.00022980575

    for i_part = 1:particle_group.n_particles
        fill!(propagator.j_dofs[1], 0.0)
        fill!(propagator.j_dofs[2], 0.0)
        xi = get_x(particle_group, i_part)
        vi = get_v(particle_group, i_part)
        s1 = get_s1(particle_group, i_part)
        s2 = get_s2(particle_group, i_part)
        s3 = get_s3(particle_group, i_part)
        wi = get_mass(particle_group, i_part)

        # Kinetic energy
        v2 = evaluate(thdiag.kernel_smoother_0, xi, afield_dofs[1])
        v3 = evaluate(thdiag.kernel_smoother_0, xi, afield_dofs[2])
        diagnostics[1] += 0.5 * (vi^2 + v2^2 + v3^2) * wi # 0.5 * wi[1] * vi[1]^2  

        #Zeeman energy
        add_charge!(propagator.j_dofs[2], propagator.kernel_smoother_1, xi, 1.0)
        compute_rderivatives_from_basis!(
            propagator.j_dofs[1],
            propagator.maxwell_solver,
            propagator.j_dofs[2],
        )
        diagnostics[2] +=
            HH * (
                afield_dofs[2]' * propagator.j_dofs[1] * wi * s2 -
                afield_dofs[1]' * propagator.j_dofs[1] * wi * s3
            )

        # Momentum: compute integrals of f like <s* f> = int s* f dx dp ds
        diagnostics[3] += xi * wi
        diagnostics[4] += xi * wi * s1
        diagnostics[5] += xi * wi * s2
        diagnostics[6] += xi * wi * s3
        diagnostics[7] += wi * s1
        diagnostics[8] += wi * s2
        diagnostics[9] += wi * s3
        diagnostics[10] += (
            afield_dofs[1]' * propagator.j_dofs[1] * wi * s2 +
            afield_dofs[2]' * propagator.j_dofs[1] * wi * s3
        )
        diagnostics[11] += afield_dofs[1]' * propagator.j_dofs[1] * wi * s1 * (-1.0)
        diagnostics[12] += afield_dofs[2]' * propagator.j_dofs[1] * wi * s1 * (-1.0)
    end

    transfer = pic_diagnostics_transfer(
        particle_group,
        thdiag.kernel_smoother_0,
        thdiag.kernel_smoother_1,
        efield_dofs,
    )

    #Energies
    #Electric energies int | E_*(x) |^2 dx

    potential_energy[1] =
        0.5 *
        inner_product(thdiag.maxwell_solver, efield_dofs[1], efield_dofs[1], degree - 1)

    potential_energy[2] =
        0.5 * inner_product(thdiag.maxwell_solver, efield_dofs[2], efield_dofs[2], degree)

    potential_energy[3] =
        0.5 * inner_product(thdiag.maxwell_solver, efield_dofs[3], efield_dofs[3], degree)

    # Magnetic energy
    nn = thdiag.kernel_smoother_0.n_dofs
    aa = zeros(nn)
    compute_lderivatives_from_basis!(aa, thdiag.maxwell_solver, afield_dofs[1])
    potential_energy[4] = 0.5 * l2norm_squared(thdiag.maxwell_solver, aa, degree - 1)
    bb = zeros(nn)
    compute_lderivatives_from_basis!(bb, thdiag.maxwell_solver, afield_dofs[2])
    potential_energy[5] = 0.5 * l2norm_squared(thdiag.maxwell_solver, bb, degree - 1)

    push!(
        thdiag.data,
        (
            time,
            diagnostics...,
            potential_energy...,
            transfer,
            maximum(abs.(efield_dofs[1] .- efield_poisson)),
        ),
    )
end


