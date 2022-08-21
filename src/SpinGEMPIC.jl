module SpinGEMPIC

using Distributed

# Utilities
include("distributions.jl")

# Particle Group
include("particle_group.jl")

# Particle-Mesh coupling
include("particle_mesh_coupling.jl")

# Particle sampling
include("particle_sampling.jl")

# Splittings
include("hamiltonian_splitting.jl")
include("hp.jl")
include("ha.jl")
include("he.jl")
include("hs.jl")

# Diagnostics
include("diagnostics.jl")

end
