<!-- #region -->
# SpinGEMPIC.jl

Geometric Particle-in-Cell methods for the Vlasov-Maxwell equations with spin effects.

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliavlasov.github.io/SpinGEMPIC.jl/dev)
[![CI](https://github.com/JuliaVlasov/SpinGEMPIC.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaVlasov/SpinGEMPIC.jl/actions/workflows/ci.yml)

## Spin Vlasov-Maxwell equations

The dimensionless non-relativistic spin Vlasov-Maxwell system is
the Vlasov equation:

$$
\frac{\partial f}{\partial t}+{\bf v}\cdot\frac{\partial f}{\partial{\bf x}}+[\left({\bf E}+{\bf v}\times{\bf B}\right) - \nabla({\bf s} \cdot {\bf B})]\cdot\frac{\partial f}{\partial{\bf v}} - ({\bf s}\times {\bf B}) \cdot \frac{\partial f}{\partial {\bf s}} = 0, 
$$

coupled with Maxwell equations.

## Installation

In a Julia session switch to `pkg>` mode to add `SpinGEMPIC`:

```julia
julia>] # switch to pkg> mode
pkg> add https://github.com/juliavlasov/SpinGEMPIC.jl
```

```julia
julia> using SpinGEMPIC
```
