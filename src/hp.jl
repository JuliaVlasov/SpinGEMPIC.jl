"""
    operatorHp(h, particle_group, dt)

```math
\\begin{aligned}
\\dot{x} & =p \\\\
\\dot{E}_x & = - \\int (p f ) dp ds
\\end{aligned}
```
"""
function operatorHp(h::HamiltonianSplitting, particle_group, dt::Float64)

    nx :: Int = h.kernel_smoother_0.n_dofs

    fill!(h.j_dofs[1], 0.0)

    @inbounds for i_part = 1:particle_group.n_particles

        # Read out particle position and velocity
        x_old = particle_group.array[1, i_part]
        vi = particle_group.array[2, i_part]

        # Then update particle position:  X_new = X_old + dt * V
        x_new = x_old + dt * vi

        # Get charge for accumulation of j
        wi = particle_group.charge * particle_group.array[6, i_part] * particle_group.common_weight
        qoverm = particle_group.q_over_m

        add_current_update_v!(
            h.j_dofs[1],
            h.kernel_smoother_1,
            x_old,
            x_new,
            wi[1],
            qoverm,
            vi,
        )

        particle_group.array[1, i_part] = mod(x_new, h.Lx)

    end

    # Update the electric field with Ampere
    compute_e_from_j!(h.e_dofs[1], h.maxwell_solver, h.j_dofs[1], 1)

end


