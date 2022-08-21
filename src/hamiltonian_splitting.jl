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
    strang_splitting( h, pg, dt, number_steps)

Strang splitting
- time splitting object 
- time step
- number of time steps
"""
function strang_splitting!(h::HamiltonianSplitting, pg::ParticleGroup, dt::Float64, number_steps::Int64)

    for i_step = 1:number_steps

        operatorHE(h, pg, 0.5dt)
        operatorHp(h, pg, 0.5dt)
        operatorHA(h, pg, 0.5dt)
        operatorHs(h, pg, 1.0dt)
        operatorHA(h, pg, 0.5dt)
        operatorHp(h, pg, 0.5dt)
        operatorHE(h, pg, 0.5dt)

    end

end

