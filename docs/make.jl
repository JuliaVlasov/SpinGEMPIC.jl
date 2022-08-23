using Documenter
using SpinGEMPIC

makedocs(
    sitename = "SpinGEMPIC",
    format = Documenter.HTML(;
    prettyurls=get(ENV, "CI", nothing) == "true",
    mathengine = MathJax3(Dict(
    :loader => Dict("load" => ["[tex]/physics"]),
    :tex => Dict(
        "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
        "tags" => "ams",
        "packages" => ["base", "ams", "autoload", "physics"],
    ),
)),
                            ),
    modules = [SpinGEMPIC],
    pages = ["Documentation" => ["index.md",
                                 "quickstart.md",
                                 "example.md",
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
