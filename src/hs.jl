"""
    operatorHs(h, particle_group, dt)

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
function operatorHs(h::HamiltonianSplitting, particle_group, dt::Float64)
    HH = 0.00022980575
    nx :: Int = h.kernel_smoother_0.n_dofs

    fill!(h.part1, 0.0)
    fill!(h.part2, 0.0)
    fill!(h.part3, 0.0)
    fill!(h.part4, 0.0)
    hat_v = zeros(3, 3)
    S = zeros(3)
    St = zeros(3)

    charge :: Float64 = particle_group.charge * particle_group.common_weight

    n_particles :: Int = particle_group.n_particles

    @inbounds for i_part = 1:n_particles

        xi :: Float64 = particle_group.array[1, i_part]
        v_new :: Float64 = particle_group.array[2, i_part]

        fill!(h.j_dofs[1], 0.0)
        fill!(h.j_dofs[2], 0.0)

        add_charge!(h.j_dofs[2], h.kernel_smoother_1, xi, 1.0)
        
        compute_rderivatives_from_basis!(h.j_dofs[1], h.maxwell_solver, h.j_dofs[2])

        Y :: Float64 = h.a_dofs[1]' * h.j_dofs[1]
        Z :: Float64 = h.a_dofs[2]' * h.j_dofs[1]

        hat_v[1, 2] = Y
        hat_v[1, 3] = Z
        hat_v[2, 1] = -Y
        hat_v[3, 1] = -Z

        s1 = particle_group.array[3, i_part]
        s2 = particle_group.array[4, i_part]
        s3 = particle_group.array[5, i_part]

        S[1] = s1
        S[2] = s2
        S[3] = s3

        vnorm = sqrt(Z*Z+Y*Y)

        α = ( sin(dt * vnorm) / vnorm * hat_v +
             0.5 * (sin(0.5dt * vnorm) / (0.5vnorm))^2 * hat_v^2)

        St .= S
        BLAS.gemm!('N','N', 1., α, S, 1., St)
        particle_group.array[3:5, i_part] .= St

        β = ( 2 * (sin(0.5dt * vnorm) / vnorm)^2 * hat_v + 2.0 / (vnorm^2) *
                    (0.5dt - sin(dt * vnorm) / 2 / vnorm) * hat_v^2) 

        St .= S
        BLAS.gemm!('N','N', 1., β, S, dt, St)

        wi = charge * particle_group.array[6, i_part] 

        # define part1 and part2
        h.part1 .+= wi * St[3] * h.j_dofs[2]
        h.part2 .-= wi * St[2] * h.j_dofs[2]

        # update velocity
        fill!(h.j_dofs[1], 0.0)
        fill!(h.j_dofs[2], 0.0)

        add_charge!(h.j_dofs[2], h.kernel_smoother_2, xi, 1.0)

        # compute rderivatives
        compute_rderivatives_from_basis!(h.j_dofs[1], h.maxwell_solver, h.j_dofs[2])
        compute_rderivatives_from_basis!(h.j_dofs[2], h.maxwell_solver, h.j_dofs[1])

        vi = v_new - HH * h.a_dofs[2]' * h.j_dofs[2] * St[2] +
                     HH * h.a_dofs[1]' * h.j_dofs[2] * St[3]

        particle_group.array[2, i_part] = vi


    end

    # Update Ey with the term  HH*int (sz df/dx) dp ds  
    # Update Ez with the term -HH*int (sy df/dx) dp ds  

    compute_rderivatives_from_basis!(h.j_dofs[1], h.maxwell_solver, h.part1)
    compute_rderivatives_from_basis!(h.j_dofs[2], h.maxwell_solver, h.part2)

    h.j_dofs[1] .*= HH
    h.j_dofs[2] .*= HH

    compute_e_from_j!(h.e_dofs[2], h.maxwell_solver, h.j_dofs[1], 2)
    compute_e_from_j!(h.e_dofs[3], h.maxwell_solver, h.j_dofs[2], 2)


end
