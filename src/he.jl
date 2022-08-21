"""
    operatorHE(h, particle_group, dt)

```math
\\begin{aligned}
\\dot{v}   & =  E_x \\\\
\\dot{A}_y & = -E_y \\\\
\\dot{A}_z & = -E_z
\\end{aligned}
```
"""
function operatorHE(h::HamiltonianSplitting, particle_group, dt::Float64)


    @inbounds for i_part = 1:particle_group.n_particles

        xi = particle_group.array[1, i_part]
        vi = particle_group.array[2, i_part]
        e1 = evaluate(h.kernel_smoother_1, xi[1], h.e_dofs[1])
        vi = vi + dt * e1

        particle_group.array[2,i_part] = vi

    end

    h.a_dofs[1] .= h.a_dofs[1] .- dt * h.e_dofs[2]
    h.a_dofs[2] .= h.a_dofs[2] .- dt * h.e_dofs[3]

end


