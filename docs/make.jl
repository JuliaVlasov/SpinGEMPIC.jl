using Documenter
using SpinGEMPIC

makedocs(
    sitename = "SpinGEMPIC",
    format = Documenter.HTML(),
    modules = [SpinGEMPIC],
    pages = ["Documentation" => ["index.md",
                                 "mesh.md",
                                 "particle_mesh_coupling.md",
                                 "distributions.md",
                                 "particle_group.md",
                                 "particle_sampling.md",
                                 "hamiltonian_splitting.md",
                                 "diagnostics.md"],
             "Contents"      => "contents.md"]
)

deploydocs(
    repo   = "github.com/JuliaVlasov/SpinGEMPIC.jl.git",
 )
