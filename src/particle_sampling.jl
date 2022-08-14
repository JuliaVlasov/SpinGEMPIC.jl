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
struct ParticleSampler{D,V,S}

    dims::Tuple{Int64,Int64,Int64}
    n_particles::Int

    function ParticleSampler{D,V,S}( n_particles::Int64,) where {D,V,S}

        dims = (D, V, S)

        new(dims, n_particles)

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

    ndx, ndv, nds = df.dims

    x = zeros(ndx)
    v = zeros(ndv)
    s = zeros(nds)
    theta = 0.0
    phi = 0.0
    n_rnds = 0
    if df.params.n_gaussians > 1
        n_rnds = 1
    end

    δ = zeros(df.params.n_gaussians)
    for i_v = 1:df.params.n_gaussians
        δ[i_v] = sum(df.params.δ[1:i_v])
    end

    n_rnds += ndx + ndv
    rdn = zeros(ndx + ndv + 1)

    # 1/Np in common weight
    set_common_weight(pg, (1.0 / pg.n_particles))

    rng_sobol = SobolSeq(ndx)

    d = Normal()

    for i_part = 1:(pg.n_particles)

        x .= mesh.xmin .+ Sobol.next!(rng_sobol) .* mesh.dimx

        # Sampling in v
        v .= rand!(rng, d, v)

        # For multiple Gaussian, draw which one to take
        rnd_no = rdn[ndx+ndv+1]

        i_gauss = 1
        while (rnd_no > δ[i_gauss])
            i_gauss += 1
        end
        v .= v .* df.params.σ[i_gauss] .+ df.params.μ[i_gauss]

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
        w = 1 / (4 * pi) * (1 + 0.5 * s[3]) .* 4 * pi

        # Set weight according to value of perturbation
        w = w * eval_x_density(df, x) .* prod(mesh.dimx)

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
