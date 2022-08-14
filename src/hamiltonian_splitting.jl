import GEMPIC: AbstractMaxwellSolver
import GEMPIC: Maxwell1DFEM
import GEMPIC: compute_e_from_rho!
import GEMPIC: compute_rderivatives_from_basis!
import GEMPIC: compute_lderivatives_from_basis!
import GEMPIC: inner_product
import GEMPIC: l2norm_squared
import GEMPIC: compute_e_from_j!
import GEMPIC: compute_e_from_b!

using StaticArrays

abstract type AbstractSplitting end

export HamiltonianSplitting


"""
    HamiltonianSplitting( maxwell_solver,
                          kernel_smoother_0, kernel_smoother_1,
                          kernel_smoother_2,
                          particle_group, e_dofs, a_dofs, domain) 

Hamiltonian splitting type for Vlasov-Maxwell

- Integral over the spline function on each interval (order p+1)
- Integral over the spline function on each interval (order p)
- `e_dofs` describing the two components of the electric field
- `b_dofs` describing the magnetic field
- `j_dofs` for kernel representation of current density. 
"""
struct HamiltonianSplitting

    maxwell_solver::AbstractMaxwellSolver
    kernel_smoother_0::ParticleMeshCoupling
    kernel_smoother_1::ParticleMeshCoupling
    kernel_smoother_2::ParticleMeshCoupling
    particle_group::ParticleGroup

    spline_degree::Int64
    Lx::Float64
    x_min::Float64
    delta_x::Float64

    cell_integrals_0::SVector
    cell_integrals_1::SVector

    e_dofs::Array{Array{Float64,1}}
    a_dofs::Array{Array{Float64,1}}
    j_dofs::Array{Array{Float64,1}}
    part1::Array{Float64,1}
    part2::Array{Float64,1}
    part3::Array{Float64,1}
    part4::Array{Float64,1}

    function HamiltonianSplitting(
        maxwell_solver,
        kernel_smoother_0,
        kernel_smoother_1,
        kernel_smoother_2,
        particle_group,
        e_dofs,
        a_dofs,
        domain::Vector{Float64},
    )

        # Check that n_dofs is the same for both kernel smoothers.
        @assert kernel_smoother_0.n_dofs == kernel_smoother_1.n_dofs

        j_dofs = [zeros(Float64, kernel_smoother_0.n_dofs) for i = 1:2]

        nx = maxwell_solver.n_dofs

        part1 = zeros(Float64, nx)
        part2 = zeros(Float64, nx)
        part3 = zeros(Float64, nx)
        part4 = zeros(Float64, nx)
        x_min = domain[1]
        Lx = domain[3]
        spline_degree = 3
        delta_x = Lx / kernel_smoother_1.n_dofs

        cell_integrals_1 = SVector{3}([0.5, 2.0, 0.5] ./ 3.0)
        cell_integrals_0 = SVector{4}([1.0, 11.0, 11.0, 1.0] ./ 24.0)

        new(
            maxwell_solver,
            kernel_smoother_0,
            kernel_smoother_1,
            kernel_smoother_2,
            particle_group,
            spline_degree,
            Lx,
            x_min,
            delta_x,
            cell_integrals_0,
            cell_integrals_1,
            e_dofs,
            a_dofs,
            j_dofs,
            part1,
            part2,
            part3,
            part4,
        )

    end

end

export strang_splitting!

"""
    strang_splitting( h, dt, number_steps)

Strang splitting
- time splitting object 
- time step
- number of time steps
"""
function strang_splitting!(h::HamiltonianSplitting, dt::Float64, number_steps::Int64)

    for i_step = 1:number_steps

        operatorHE(h, 0.5dt)
        operatorHp(h, 0.5dt)
        operatorHA(h, 0.5dt)
        operatorHs(h, 1.0dt)
        operatorHA(h, 0.5dt)
        operatorHp(h, 0.5dt)
        operatorHE(h, 0.5dt)

    end

end

"""
    operatorHp(h, dt)
```math
\\begin{aligned}
\\dot{x} & =p \\\\
\\dot{E}_x & = - \\int (p f ) dp ds
\\end{aligned}
```
"""

function operatorHp(h::HamiltonianSplitting, dt::Float64)

    n_cells = h.kernel_smoother_0.n_dofs

    fill!(h.j_dofs[1], 0.0)
    fill!(h.j_dofs[2], 0.0)

    for i_part = 1:h.particle_group.n_particles

        # Read out particle position and velocity
        x_old = get_x(h.particle_group, i_part)
        vi = get_v(h.particle_group, i_part)

        # Then update particle position:  X_new = X_old + dt * V
        x_new = x_old .+ dt * vi

        # Get charge for accumulation of j
        wi = get_charge(h.particle_group, i_part)
        qoverm = h.particle_group.q_over_m

        add_current_update_v!(
            h.j_dofs[1],
            h.kernel_smoother_1,
            x_old,
            x_new,
            wi[1],
            qoverm,
            vi,
        )

        x_new = mod(x_new, h.Lx)
        set_x(h.particle_group, i_part, x_new)

    end

    # Update the electric field with Ampere
    compute_e_from_j!(h.e_dofs[1], h.maxwell_solver, h.j_dofs[1], 1)

end


"""
    operatorHA(h, dt)
```math
\\begin{aligned}
\\dot{p} = (A_y, A_z) \\cdot \\partial_x (A_y, A_z)   \\\\
\\dot{Ey} = -\\partial_x^2 A_y + A_y \\rho \\\\
\\dot{Ez} = -\\partial_x^2 A_z + A_z \\rho \\\\
\\end{aligned}
```
"""

function operatorHA(h::HamiltonianSplitting, dt::Float64)

    n_cells = h.kernel_smoother_0.n_dofs
    fill!(h.part1, 0.0)
    fill!(h.part2, 0.0)
    fill!(h.part3, 0.0)
    fill!(h.part4, 0.0)
    aa = zeros(Float64, n_cells)


    qm = h.particle_group.q_over_m
    # Update v_1
    for i_part = 1:h.particle_group.n_particles

        fill!(h.j_dofs[1], 0.0)
        fill!(h.j_dofs[2], 0.0)

        xi = get_x(h.particle_group, i_part)
        vi = get_v(h.particle_group, i_part)
        wi = get_charge(h.particle_group, i_part)

        add_charge!(h.j_dofs[2], h.kernel_smoother_0, xi, 1.0)
        add_charge!(h.j_dofs[1], h.kernel_smoother_1, xi, 1.0)

        # values of the derivatives of basis function
        compute_rderivatives_from_basis!(aa, h.maxwell_solver, h.j_dofs[1])
        h.j_dofs[1] .= aa

        vi = vi - dt / 2 * (h.a_dofs[1]' * h.j_dofs[1] * (h.j_dofs[2]' * h.a_dofs[1]))
        vi = vi - dt / 2 * (h.a_dofs[1]' * h.j_dofs[2] * (h.j_dofs[1]' * h.a_dofs[1]))
        vi = vi - dt / 2 * (h.a_dofs[2]' * h.j_dofs[1] * (h.j_dofs[2]' * h.a_dofs[2]))
        vi = vi - dt / 2 * (h.a_dofs[2]' * h.j_dofs[2] * (h.j_dofs[1]' * h.a_dofs[2]))

        set_v(h.particle_group, i_part, vi)

        # below we solve electric field
        # first define part1 and part2 to be 0 vector
        h.part1 .= h.part1 .+ dt * wi * (h.j_dofs[2]' * h.a_dofs[1]) * h.j_dofs[2]
        h.part2 .= h.part2 .+ dt * wi * (h.j_dofs[2]' * h.a_dofs[2]) * h.j_dofs[2]


    end

    # Update the electric field 
    # with the (A rho) part 

    compute_e_from_j!(h.e_dofs[2], h.maxwell_solver, -h.part1, 2)
    compute_e_from_j!(h.e_dofs[3], h.maxwell_solver, -h.part2, 2)

    # with the (d^2 A/ dx^2) part 

    compute_lderivatives_from_basis!(h.part3, h.maxwell_solver, h.a_dofs[1])
    compute_lderivatives_from_basis!(h.part4, h.maxwell_solver, h.a_dofs[2])

    compute_e_from_b!(h.e_dofs[2], h.maxwell_solver, dt, h.part3)
    compute_e_from_b!(h.e_dofs[3], h.maxwell_solver, dt, h.part4)


end


"""
    operatorHE(h, dt)
```math
\\begin{aligned}
\\dot{v}   & =  E_x \\\\
\\dot{A}_y & = -E_y \\\\
\\dot{A}_z & = -E_z
\\end{aligned}
```
"""

function operatorHE(h::HamiltonianSplitting, dt::Float64)


    for i_part = 1:h.particle_group.n_particles

        vi = get_v(h.particle_group, i_part)
        xi = get_x(h.particle_group, i_part)
        e1 = evaluate(h.kernel_smoother_1, xi[1], h.e_dofs[1])
        vi = vi + dt * e1

        set_v(h.particle_group, i_part, vi)

    end

    h.a_dofs[1] .= h.a_dofs[1] .- dt * h.e_dofs[2]
    h.a_dofs[2] .= h.a_dofs[2] .- dt * h.e_dofs[3]

end


"""
    operatorHs(h, dt)
Push H_s: Equations to be solved
```math
\\begin{aligned}
\\dot{s} &= s x B = (s_y \\partial_x A_y +s_z \\partial_x Az, -s_x \\partial_x A_y, -s_x \\partial_x A_z)  \\\\
\\dot{p} &= s \\cdot \\partial_x B = -s_y \\partial^2_{x} A_z + s_z \\partial^2_{x} A_y \\\\
\\dot{E}_y &=   HH \\int (s_z \\partial_x f) dp ds \\\\
\\dot{E}_z &= - HH \\int (s_y \\partial_x f) dp ds 
\\end{aligned}
```
"""

function operatorHs(h::HamiltonianSplitting, dt::Float64)
    HH = 0.00022980575
    n_cells = h.kernel_smoother_0.n_dofs
    fill!(h.part1, 0.0)
    fill!(h.part2, 0.0)
    fill!(h.part3, 0.0)
    fill!(h.part4, 0.0)
    hat_v = zeros(Float64, 3, 3)
    S = zeros(Float64, 3)
    St = zeros(Float64, 3)
    aa = zeros(Float64, n_cells)
    bb = zeros(Float64, n_cells)


    for i_part = 1:h.particle_group.n_particles

        v_new = get_v(h.particle_group, i_part)

        # Evaluate efields at particle position
        xi = get_x(h.particle_group, i_part)
        fill!(h.j_dofs[1], 0.0)
        fill!(h.j_dofs[2], 0.0)

        add_charge!(h.j_dofs[2], h.kernel_smoother_1, xi, 1.0)

        compute_rderivatives_from_basis!(h.j_dofs[1], h.maxwell_solver, h.j_dofs[2])

        Y = h.a_dofs[1]' * h.j_dofs[1]
        Z = h.a_dofs[2]' * h.j_dofs[1]
        V = [0, Z, -Y]

        hat_v[1, 2] = Y
        hat_v[1, 3] = Z
        hat_v[2, 1] = -Y
        hat_v[3, 1] = -Z

        s1 = get_s1(h.particle_group, i_part)
        s2 = get_s2(h.particle_group, i_part)
        s3 = get_s3(h.particle_group, i_part)

        vnorm = norm(V)

        if vnorm > 1e-14
            S .=
                [s1, s2, s3] .+
                (
                    sin(dt * norm(V)) / norm(V) * hat_v +
                    0.5 * (sin(dt / 2 * norm(V)) / (norm(V) / 2))^2 * hat_v^2
                ) * [s1, s2, s3]
        else
            S .= [s1, s2, s3]
        end

        set_s1(h.particle_group, i_part, S[1])
        set_s2(h.particle_group, i_part, S[2])
        set_s3(h.particle_group, i_part, S[3])

        if vnorm > 1e-14
            St .=
                dt * [s1, s2, s3] .+
                (
                    2 * (sin(dt * norm(V) / 2) / norm(V))^2 * hat_v +
                    2.0 / (norm(V)^2) *
                    (dt / 2 - sin(dt * norm(V)) / 2 / norm(V)) *
                    hat_v^2
                ) * [s1, s2, s3]
        else
            St .= dt * [s1, s2, s3]
        end

        wi = get_charge(h.particle_group, i_part)

        # define part1 and part2
        h.part1 .= h.part1 .+ wi[1] * St[3] * h.j_dofs[2]
        h.part2 .= h.part2 .+ wi[1] * St[2] * h.j_dofs[2]

        # update velocity
        fill!(h.j_dofs[2], 0.0)
        add_charge!(h.j_dofs[2], h.kernel_smoother_2, xi, 1.0)
        compute_rderivatives_from_basis!(h.j_dofs[1], h.maxwell_solver, h.j_dofs[2])
        compute_rderivatives_from_basis!(aa, h.maxwell_solver, h.j_dofs[1])
        h.j_dofs[1] .= aa

        vi =
            v_new - HH * h.a_dofs[2]' * h.j_dofs[1] * St[2] +
            HH * h.a_dofs[1]' * h.j_dofs[1] * St[3]

        set_v(h.particle_group, i_part, vi)


    end

    # Update Ey with the term  HH*int (sz df/dx) dp ds  
    # Update Ez with the term -HH*int (sy df/dx) dp ds  

    compute_rderivatives_from_basis!(aa, h.maxwell_solver, h.part1)
    compute_rderivatives_from_basis!(bb, h.maxwell_solver, -h.part2)

    compute_e_from_j!(h.e_dofs[2], h.maxwell_solver, HH .* aa, 2)
    compute_e_from_j!(h.e_dofs[3], h.maxwell_solver, HH .* bb, 2)

end
