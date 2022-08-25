using Sobol, Random, Distributions
import LinearAlgebra: norm

export sample!

"""
    sample!( rng, pg, ps, df, mesh)

Sample from a Particle sampler

- `rnd`  : Random generator 
- `pg`   : Particle group
- `df`   : Distribution function
- `mesh` : Domain
"""
function sample!(rng, pg, df::AbstractCosGaussian, mesh)

    s = zeros(3)
    theta = 0.0
    phi = 0.0

    # 1/Np in common weight
    set_common_weight(pg, (1.0 / pg.n_particles))

    rng_sobol = SobolSeq(1)

    σ, μ = df.params.σ[1][1], df.params.μ[1][1]
    d = Normal(μ, σ)

    for i_part = 1:pg.n_particles

        x = mesh.xmin + Sobol.next!(rng_sobol)[1] * mesh.dimx
        v = rand(rng, d)

        randn!(rng, s)
        s .= s ./ norm(s)
        w = 1 + 0.5 * s[3]

        # Set weight according to value of perturbation
        w = w * eval_x_density(df, x) * mesh.dimx

        # Copy the generated numbers to the particle
        set_x(pg, i_part, x)
        set_v(pg, i_part, v)
        set_s1(pg, i_part, s[1])
        set_s2(pg, i_part, s[2])
        set_s3(pg, i_part, s[3])
        # Set weights.
        set_weights(pg, i_part, w)

    end

end
