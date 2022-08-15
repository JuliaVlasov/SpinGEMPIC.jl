export ParticleGroup

abstract type AbstractParticleGroup end

"""

    ParticleGroup( n_particles, charge, mass, n_weights)

- `n_particles` : number of particles 
- `charge`      : charge of the particle species
- `mass`        : mass of the particle species
- `n_weights`   : number of differents weights
"""
mutable struct ParticleGroup <: AbstractParticleGroup

    dims::Tuple{Int64,Int64,Int64}
    n_particles::Int64
    array::Array{Float64,2}
    common_weight::Float64
    charge::Float64
    mass::Float64
    n_weights::Int64
    q_over_m::Float64

    function ParticleGroup(n_particles, charge, mass, n_weights)

        dims = (1, 1, 3)
        array = zeros(Float64, (sum(dims) + n_weights, n_particles))
        common_weight = 1.0
        q_over_m = charge / mass

        new(
            dims,
            n_particles,
            array,
            common_weight,
            charge,
            mass,
            n_weights,
            q_over_m,
        )
    end
end

export get_x

"""  
    get_x( p, i )
Get position of ith particle of p
"""
@inline get_x(p::ParticleGroup, i::Int) = p.array[1, i]

export get_v

"""  
    get_v( p, i )
Get velocity of ith particle of p
"""
@inline get_v(p::ParticleGroup, i::Int) = p.array[2, i]


"""  
get_s1( p, i )

Get s1 of ith particle of p
"""
@inline get_s1(p::ParticleGroup, i::Int) = p.array[3, i]

"""  
get_s2( p, i )

Get s2 of ith particle of p
"""
@inline get_s2(p::ParticleGroup, i::Int) = p.array[4, i]

"""  
get_s3( p, i )

Get velocity of ith particle of p
"""
@inline get_s3(p::ParticleGroup, i::Int64) = p.array[5, i]


"""
    get_charge( p, i; i_wi=1)

Get charge of ith particle of p (q * particle_weight)
"""
@inline function get_charge(p::ParticleGroup, i::Int64; i_wi = 1)

    p.charge * p.array[5+i_wi, i] * p.common_weight

end


"""
    get_mass( p, i; i_wi=1)

Get mass of ith particle of p (m * particle_weight)
"""
@inline function get_mass(p::ParticleGroup, i::Int64; i_wi = 1)

    p.mass * p.array[5+i_wi, i] * p.common_weight

end

"""
    get_weights( p, i)

Get ith particle weights of group p
"""
@inline function get_weights(p::ParticleGroup, i::Int64) 

    p.array[6, i]

end

"""
    set_x( p, i, x ) 

Set position of ith particle of p to x 
"""
@inline function set_x( p::ParticleGroup, i::Int64, x::Float64)

    p.array[1, i] = x

end

"""
    set_v( p, i, v)

Set velocity of ith particle of p to v
"""
@inline function set_v(p::ParticleGroup, i::Int64, v::Float64) 

    p.array[2, i] = v

end

"""
set_s1( p, i, v)

Set velocity of ith particle of p to v
"""
@inline function set_s1(p::ParticleGroup, i::Int64, v::Float64) 

    p.array[3, i] = v

end


"""
set_s2( p, i, v)

Set velocity of ith particle of p to v
"""
@inline function set_s2(p::ParticleGroup, i::Int64, v::Float64)

    p.array[4, i] = v

end

"""
set_s3( p, i, v)

Set velocity of ith particle of p to v
"""
@inline function set_s3(p::ParticleGroup, i::Int64, v::Float64) 

    p.array[5, i] = v

end


"""
    set_weights( p, i, w) 

Set weights of particle @ i
"""
@inline function set_weights(p::ParticleGroup, i::Int64, w::Float64) 

    p.array[6, i] = w

end

export set_common_weight
"""
    set_common_weight( p, x ) 

Set the common weight
"""
function set_common_weight(p::AbstractParticleGroup, x::Float64)

    p.common_weight = x

end
