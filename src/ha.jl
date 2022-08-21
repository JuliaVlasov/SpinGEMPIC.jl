"""
    operatorHA(h, particle_group, dt)

```math
\\begin{aligned}
\\dot{p} = (A_y, A_z) \\cdot \\partial_x (A_y, A_z)   \\\\
\\dot{Ey} = -\\partial_x^2 A_y + A_y \\rho \\\\
\\dot{Ez} = -\\partial_x^2 A_z + A_z \\rho \\\\
\\end{aligned}
```
"""
function operatorHA(h::HamiltonianSplitting, particle_group, dt::Float64)

    nx :: Int = h.kernel_smoother_0.n_dofs
    fill!(h.part1, 0.0)
    fill!(h.part2, 0.0)
    fill!(h.part3, 0.0)
    fill!(h.part4, 0.0)
    aa = zeros(Float64, nx)

    charge = particle_group.charge * particle_group.common_weight

    @inbounds for i_part = 1:particle_group.n_particles

        fill!(h.j_dofs[1], 0.0)
        fill!(h.j_dofs[2], 0.0)

        xi = particle_group.array[1, i_part]
        vi = particle_group.array[2, i_part]
        wi = charge * particle_group.array[6, i_part] 

        add_charge!(h.j_dofs[2], h.kernel_smoother_0, xi, 1.0)
        add_charge!(h.j_dofs[1], h.kernel_smoother_1, xi, 1.0)

        # values of the derivatives of basis function
        compute_rderivatives_from_basis!(aa, h.maxwell_solver, h.j_dofs[1])
        h.j_dofs[1] .= aa

        vi -= 0.5dt * (h.a_dofs[1]' * h.j_dofs[1] * (h.j_dofs[2]' * h.a_dofs[1]))
        vi -= 0.5dt * (h.a_dofs[1]' * h.j_dofs[2] * (h.j_dofs[1]' * h.a_dofs[1]))
        vi -= 0.5dt * (h.a_dofs[2]' * h.j_dofs[1] * (h.j_dofs[2]' * h.a_dofs[2]))
        vi -= 0.5dt * (h.a_dofs[2]' * h.j_dofs[2] * (h.j_dofs[1]' * h.a_dofs[2]))

        particle_group.array[2,i_part] = vi

        # below we solve electric field
        # first define part1 and part2 to be 0 vector
        h.part1 .-= dt * wi * (h.j_dofs[2]' * h.a_dofs[1]) * h.j_dofs[2]
        h.part2 .-= dt * wi * (h.j_dofs[2]' * h.a_dofs[2]) * h.j_dofs[2]


    end

    # Update the electric field with the (A rho) part 

    compute_e_from_j!(h.e_dofs[2], h.maxwell_solver, h.part1, 2)
    compute_e_from_j!(h.e_dofs[3], h.maxwell_solver, h.part2, 2)

    # with the (d^2 A/ dx^2) part 

    compute_lderivatives_from_basis!(h.part3, h.maxwell_solver, h.a_dofs[1])
    compute_lderivatives_from_basis!(h.part4, h.maxwell_solver, h.a_dofs[2])

    compute_e_from_b!(h.e_dofs[2], h.maxwell_solver, dt, h.part3)
    compute_e_from_b!(h.e_dofs[3], h.maxwell_solver, dt, h.part4)


end


