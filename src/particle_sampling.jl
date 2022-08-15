using Sobol, Random, Distributions
using LinearAlgebra

export ParticleSampler

"""
    ParticleSampler{D,V,S}( sampling_type, symmetric, dims, n_particles)

Particle initializer class with various functions to initialize a particle.

- `sampling_type` : `:random` or `:sobol`
- `symmetric` : `true` or `false`
- `n_particles` : number of particles
"""
struct ParticleSampler

    n_particles::Int

    function ParticleSampler( n_particles::Int64)

        new(n_particles)

    end

end

export sample!

"""
    sample!( rng, pg, ps, df, mesh)

Sample from a Particle sampler

- `pg`   : Particle group
- `ps`   : Particle sampler
- `df`   : Distribution function
- `xmin` : lower bound of the domain
- `Lx`   : length of the domain.
"""
function sample!(rng, pg, ps :: ParticleSampler, df::AbstractCosGaussian, mesh)

    s = zeros(3)
    theta = 0.0
    phi = 0.0

    # 1/Np in common weight
    set_common_weight(pg, (1.0 / pg.n_particles))

    rng_sobol = SobolSeq(1)

    d = Normal()

    for i_part = 1:pg.n_particles

        x = mesh.xmin + Sobol.next!(rng_sobol)[1] * mesh.dimx

        # Sampling in v
        v = rand(rng, d)

        v = v * df.params.σ[1][1] + df.params.μ[1][1]

        # Wigner type initial condition 1/(4pi) (1+eta*s[3]) fM(v)
        for tt = 1:10
            s[1] = randn(rng)
            s[2] = randn(rng)
            s[3] = randn(rng)
            if norm(s) > 10^(-4)
                break
            end
        end
        s .= s ./ norm(s)
        w = 1 / (4pi) * (1 + 0.5 * s[3]) .* 4pi

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
