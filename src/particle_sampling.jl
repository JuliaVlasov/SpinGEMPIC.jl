using Sobol, Random, Distributions
import LinearAlgebra: norm

export sample!

"""
    sample!( rng, pg, df, mesh, method)

Sample from a Particle sampler

- `rnd`  : Random generator 
- `pg`   : Particle group
- `df`   : Distribution function
- `mesh` : Domain
- `method` : `weighted` or `quietstart` with weigth = 1
"""
function sample!(rng, pg, df::AbstractCosGaussian, mesh; method = :weighted)

    if method == :weighted
         sample_weighted!(rng, pg, df, mesh)
    elseif method == :quietstart
         sample_quietstart!(rng, pg, df, mesh)
    else
        @error("Choose your sampling method :weighted or :quietstart")
    end

end

"""
    sample_weighted!( rng, pg, df, mesh)

Sample from a Particle sampler

- `rnd`  : Random generator 
- `pg`   : Particle group
- `df`   : Distribution function
- `mesh` : Domain
"""
function sample_weighted!(rng, pg, df, mesh)

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

"""
    sample_quietsart!( rng, pg, df, mesh)

Sample from a Particle sampler

- `rnd`  : Random generator 
- `pg`   : Particle group
- `df`   : Distribution function
- `mesh` : Domain
"""
function sample_quietstart!(rng, pg, df, mesh)

    set_common_weight(pg, (1.0 / pg.n_particles))

    r_sobol = SobolSeq(2)
    s_sobol = SobolSeq(2)

    σ, μ = df.params.σ[1][1], df.params.μ[1][1]
    α, kx = df.params.α[1], df.params.k[1][1]

    n = pg.n_particles

    # Cumulative distribution function 
    xmin, xmax = mesh.xmin, mesh.xmax
    x = LinRange(xmin, xmax, n) |> collect
    f = 1 .+ α * cos.(kx .* x)
    dx = (xmax - xmin) / (n-1)
    cdf_f = cumsum(f) * dx 

    y = LinRange(-1, 1, n) |> collect
    g = 1 .+ y ./ 2
    dy = 2.0 / (n-1)
    cdf_g = cumsum(g) * dy 

    for i_part = 1:n

        # x, v
        r1, r2 = Sobol.next!(r_sobol)
        v = σ * sqrt(-2 * log( (i_part-0.5)/n)) * sin(2π * r2)

        i = findmin(abs.(cdf_f .- r1 * mesh.dimx) )[2]

        # s1, s2, s3
        z1, z2 = Sobol.next!(s_sobol)
        j = findmin(abs.(cdf_g .- 2z1))[2]

        s3 = y[j]
        θ = 4π * z2 

        s1 = sin(θ) * sqrt(1-s3^2)
        s2 = cos(θ) * sqrt(1-s3^2)

        w = mesh.dimx

        # Copy the generated numbers to the particle
        set_x(pg, i_part, x[i])
        set_v(pg, i_part, v)
        set_s1(pg, i_part, s1)
        set_s2(pg, i_part, s2)
        set_s3(pg, i_part, s3)
        # Set weights.
        set_weights(pg, i_part, w)

    end

end
