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

Input r is a random number ``\\in [0,1]``

```math
    f(x) = 1 + \\alpha cos(k x)
```
on some domain ``[0, 2\\pi/k]``

Solve the equation ``P(x)-r=0`` with Newton’s method

```math
    x^{n+1} = x^n – (P(x)-(2\\pi r / k)/f(x) 
```

with 
```math
P(x) = \\int_0^x (1 + \\alpha cos(k y)) dy
```
```math
P(x) = x + \\frac{\\alpha}{k} sin (k x)
```
"""
function sample_quietstart!(rng, pg, df, mesh)

    set_common_weight(pg, 1.0)

    r_sobol = SobolSeq(2)
    s_sobol = SobolSeq(2)

    σ, μ = df.params.σ[1][1], df.params.μ[1][1]
    α, kx = df.params.α[1], df.params.k[1][1]

    n = pg.n_particles

    # Cumulative distribution function 
    xmin, xmax = mesh.xmin, mesh.xmax

    function newton_x(r)
        x0, x1 = xmin, xmax
        r *= (xmax - xmin)
        while (abs(x1-x0) > 1e-12)
            p = x0 + α * sin( kx * x0) / kx
            f = 1 + α * cos( kx * x0)
            x0, x1 = x1, x0 - (p - r) / f
        end
        x1
    end

    function newton_s3(r)
        x0, x1 = -1, 1
        s = -3/4 + 2r 
        while (abs(x1-x0) > 1e-12)
            p = x0 * ( 1 + x0 / 4)
            f = 1 + x0 / 2
            x0, x1 = x1, x0 - (p - s) / f
        end
        x1
    end

    for i_part = 1:n

        # x, v
        r1, r2 = Sobol.next!(r_sobol)

        x = newton_x(r1)
        v = σ * sqrt(-2 * log( (i_part-0.5)/n)) * sin(2π * r2)

        # s1, s2, s3
        z1, z2 = Sobol.next!(s_sobol)

        s3 = newton_s3(z1)
        θ = 4π * z2 

        s1 = sin(θ) * sqrt(1-s3^2)
        s2 = cos(θ) * sqrt(1-s3^2)

        w = mesh.dimx / n

        # Copy the generated numbers to the particle
        set_x(pg, i_part, x)
        set_v(pg, i_part, v)
        set_s1(pg, i_part, s1)
        set_s2(pg, i_part, s2)
        set_s3(pg, i_part, s3)
        # Set weights.
        set_weights(pg, i_part, w)

    end

end
