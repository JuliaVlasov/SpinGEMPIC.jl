# Hamiltonian splitting

```@autodocs
Modules = [SpinGEMPIC]
Pages   = ["hamiltonian_splitting.jl","hp.jl","hs.jl","ha.jl","he.jl"]
Private  = false
```

## Subsystem corresponding to ``H_p``

```math
H_p = \frac{1}{2}{\mathbf P}^{\mathrm{T}}\mathbb{M}_p{\mathbf P}
```

```@docs
SpinGEMPIC.operatorHp
```

```math
\begin{equation}
\begin{aligned}
&\dot{\mathbf X} = {\mathbf P},\\
&\dot{\mathbf P} = {\mathbf 0},\\
&\dot{\mathbf S} = {\mathbf 0},\\
&\dot{\mathbf e}_x = -\mathbb{M}_1^{-1}\mathbb{R}^1({\mathbf X})^{\mathrm{T}}\mathbb{M}_p{\mathbf P}\\
&\dot{\mathbf e}_y = {\mathbf 0},\\
&\dot{\mathbf e}_z = {\mathbf 0},\\
&\dot{\mathbf a}_y = {\mathbf 0},\\
&\dot{\mathbf a}_z = {\mathbf 0}.
\end{aligned}
\end{equation}
```

For this subsystem, we only need to compute `` {\mathbf X}, {\mathbf{e}}_x``.

```math
\begin{equation}
\begin{aligned}
&{\mathbf X}(t) = {\mathbf X}(0) + t {\mathbf P}(0),\\
&\mathbb{M}_1{\mathbf{e}}_x(t) = \mathbb{M}_1{\mathbf{e}}_x(0) - \int_0^t \mathbb{R}^1({\mathbf X}(\tau))^{\mathrm{T}}\mathbb{M}_p{\mathbf P}\mathrm{d}\tau.
\end{aligned}
\end{equation}
```

## Subsystem corresponding to ``H_A``

```@docs
SpinGEMPIC.operatorHA
```

```math
H_A = \frac{1}{2}\sum_{a=1}^{N_p}\omega_a{\mathbf a_y}^{\mathrm{T}}\mathbb{N}^0(x_a){\mathbf a}_y + \sum_{a=1}^{N_p}\omega_a{\mathbf a_z}^{\mathrm{T}}\mathbb{N}^0(x_a){\mathbf a}_z + \frac{1}{2}{\mathbf a_y}^{\mathrm{T}}\mathbb{C}^{\mathrm{T}}{\mathbb{M}}_1\mathbb{C}{\mathbf a}_y
+ \frac{1}{2}{\mathbf a_z}^{\mathrm{T}}\mathbb{C}^{\mathrm{T}}{\mathbb{M}}_1\mathbb{C}{\mathbf a}_z.
```

```math
\begin{equation}
\begin{aligned}
&\dot{\mathbf X} = {\mathbf 0},\\
&\dot{\mathbf P}  = \mathbb{M}_p^{-1}\frac{\partial H_A}{\partial {\mathbf X}},\\
&\dot{\mathbf S} = {\mathbf  0},\\
&\dot{\mathbf e}_x = {\mathbf 0},\\
&\dot{\mathbf e}_y = \mathbb{M}_0^{-1}\left(\sum_{a=1}^{N_p}\omega_a\mathbb{N}^0(x_a){\mathbf a}_y + \mathbb{C}^{\mathrm{T}}\mathbb{M}_1\mathbb{C}{\mathbf a}_y  \right),\\
&\dot{\mathbf e}_z = \mathbb{M}_0^{-1}\left(\sum_{a=1}^{N_p}\omega_a\mathbb{N}^0(x_a){\mathbf a}_z + \mathbb{C}^{\mathrm{T}}\mathbb{M}_1\mathbb{C}{\mathbf a}_z  \right),\\
&\dot{\mathbf a}_y=  {\mathbf 0},\\
&\dot{\mathbf a}_z=  {\mathbf 0}.
\end{aligned}
\end{equation}
```

In this subsystem, ``{\mathbf X}, {\mathbf S}, {\mathbf e}_x, {\mathbf a}_y, {\mathbf a}_z`` stay unchanged along the time. As for ``{\mathbf P}``, we have 


```math
\begin{equation}
\begin{aligned}
{p_a}(t) &= {p_a}(0) + t \frac{1}{\omega_a}\left( \frac{1}{2}\omega_a {\mathbf a}_y^{\mathrm{T}}\frac{\partial }{\partial x_a}\mathbb{N}^0(x_a){\mathbf a}_y  + \frac{1}{2}\omega_a {\mathbf a}_z^{\mathrm{T}}\frac{\partial }{\partial x_a}\mathbb{N}^0(x_a){\mathbf a}_z\right),\\
&= p_a(0) + t \left( \frac{1}{2} {\mathbf a}_y^{\mathrm{T}}\frac{\partial }{\partial x_a}\mathbb{N}^0(x_a){\mathbf a}_y  + \frac{1}{2} {\mathbf a}_z^{\mathrm{T}}\frac{\partial }{\partial x_a}\mathbb{N}^0(x_a){\mathbf a}_z\right),\\
& = p_a(0) + \frac{t}{2}   {\mathbf a}_y^{\mathrm{T}}(\frac{\partial}{\partial x_a}\Lambda^0(x_a) \Lambda^0(x_a)^{\mathrm{T}}  + \Lambda^0(x_a) \frac{\partial}{\partial x_a}\Lambda^0(x_a)^{\mathrm{T}}  ){\mathbf a}_y,\\
&+ \frac{t}{2} {\mathbf a}_z^{\mathrm{T}}(\frac{\partial}{\partial x_a}\Lambda^0(x_a) \Lambda^0(x_a)^{\mathrm{T}}  + \Lambda^0(x_a) \frac{\partial}{\partial x_a}\Lambda^0(x_a)^{\mathrm{T}}  ){\mathbf a}_z.
\end{aligned}
\end{equation}
```

```math
\begin{equation}
\begin{aligned}
\mathbb{M}_0{\mathbf e}_y(t) &= \mathbb{M}_0{\mathbf e}_y(0)  + t\left(\sum_{a=1}^{N_p}\omega_a\mathbb{N}^0(x_a){\mathbf a}_y + \mathbb{C}^{\mathrm{T}}\mathbb{M}_1\mathbb{C}{\mathbf a}_y  \right),\\
& = \mathbb{M}_0{\mathbf e}_y(0)  + t\left(\sum_{a=1}^{N_p}\omega_a\Lambda^0(x_a)\Lambda^0(x_a)^{\mathrm{T}}{\mathbf a}_y + \mathbb{C}^{\mathrm{T}}\mathbb{M}_1\mathbb{C}{\mathbf a}_y  \right).
\end{aligned}
\end{equation}
```

```math
\begin{equation}
\begin{aligned}
\mathbb{M}_0{\mathbf e}_z(t) &= \mathbb{M}_0{\mathbf e}_z(0)  + t\left(\sum_{a=1}^{N_p}\omega_a\mathbb{N}^0(x_a){\mathbf a}_z + \mathbb{C}^{\mathrm{T}}\mathbb{M}_1\mathbb{C}{\mathbf a}_z  \right),\\
& = \mathbb{M}_0{\mathbf e}_z(0)  + t\left(\sum_{a=1}^{N_p}\omega_a\Lambda^0(x_a)\Lambda^0(x_a)^{\mathrm{T}}{\mathbf a}_z + \mathbb{C}^{\mathrm{T}}\mathbb{M}_1\mathbb{C}{\mathbf a}_z  \right).
\end{aligned}
\end{equation}
```

Note that in the above, we use the identity 
```math
\mathbb{N}^0(x_a) = \Lambda^0(x_a) \Lambda^0(x_a)^{\mathrm{T}},
```
and 

```math
\frac{\mathrm{d}}{\mathrm{d}x}\mathbb{N}^0(x_a)  =  \frac{\partial}{\partial x_a}\Lambda^0(x_a) \Lambda^0(x_a)^{\mathrm{T}}  + \Lambda^0(x_a) \frac{\partial}{\partial x_a}\Lambda^0(x_a)^{\mathrm{T}},
```

which reduces the computational cost.

## Subsystem corresponding to ``H_s``

```@docs
SpinGEMPIC.operatorHs
```

```math
H_s = {\mathbf a}_z^{\mathrm{T}}\mathbb{C}^{\mathrm{T}}{\mathbb{R}}^1({\mathbf{X}})^{\mathrm{T}}\mathbb{M}_p {\mathbf S}_2 -{\mathbf a}_y^{\mathrm{T}}\mathbb{C}^{\mathrm{T}}{\mathbb{R}}^1({\mathbf{X}})^{\mathrm{T}}\mathbb{M}_p {\mathbf S}_3
```

is

```math
\begin{equation}
\begin{aligned}
&\dot{\mathbf{X}} = {\mathbf 0},\\
&\dot{\mathbf P} = -\mathbb{M}_p^{-1}\frac{\partial H_{s}}{\partial {\mathbf X}},\\
&\dot{\mathbf S} = \mathbb{S}\frac{\partial H_{s}}{\partial {\mathbf S}},\\
&\dot{\mathbf e}_x = {\mathbf 0},\\
&\dot{\mathbf e}_y = - \mathbb{M}_0^{-1}\mathbb{C}^{\mathrm{T}}\mathbb{R}^{1}({\mathbf X})^{\mathrm{T}}\mathbb{M}_p{\mathbf{S}}_3,\\
&\dot{\mathbf e}_z = \mathbb{M}_0^{-1}\mathbb{C}^{\mathrm{T}}\mathbb{R}^{1}({\mathbf X})^{\mathrm{T}}\mathbb{M}_p{\mathbf{S}}_2,\\
&\dot{a}_y = {\mathbf 0},\\
&\dot{a}_z = {\mathbf 0}.
\end{aligned}
\end{equation}
```

For this subsystem, we firstly solve ``\dot{\mathbf S} = \mathbb{S}\frac{\partial H_{s}}{\partial {\mathbf S}}``. For each particle, we have

```math
\begin{align}
\begin{aligned}
\dot{s}_a = 
& \left(
\begin{matrix}
    \dot{s}_{a,1}  \\
    \dot{s}_{a,2}  \\
      \dot{s}_{a,3}  
      \end{matrix}
\right) = 
\left(
\begin{matrix}
    0 & Y& Z \\
    -Y  & 0  & 0 \\
      -Z & 0 &0 
      \end{matrix}
\right)
 \left(
\begin{matrix}
    {s}_{a,1}  \\
    {s}_{a,2}  \\
     {s}_{a,3}  
      \end{matrix}
\right) = \hat{v} {\mathbf s}_a,
\end{aligned}
\end{align}
```

where ``Y = {\mathbf a}_y^{\mathrm{T}}\mathbb{C}^{\mathrm{T}}R^1(x_a)``, ``Z = {\mathbf a}_z^{\mathrm{T}}\mathbb{C}^{\mathrm{T}}R^1(x_a)``, ``R^1(x_a) = (\Lambda^1_1(x_a), \cdots, \Lambda^1_{N_1}(x_a))^{\mathrm{T}}``.
Set a vector ``{\mathbf v}  = (0, Z, -Y) \in \mathbb{R}^3``. Then 

```math
{\mathbf s}_a(t) = \exp(t\hat{v}){\mathbf s}_a(0) = \left(I + \frac{\sin(t|{\mathbf v}|)}{|\mathbf {v}|}\hat{v} + \frac{1}{2}\left( \frac{\sin(\frac{t}{2}|{\mathbf v}|)}{\frac{|{\mathbf v}|}{2}} \right)^2 \hat{v}^2\right) {\mathbf s}_a(0),
```

and

```math
\begin{equation}
\begin{aligned}
\int_0^t {\mathbf s}_a(\tau){\mathrm{d}}\tau= \int_0^t \exp(\tau \hat{v}){\mathbf s}_a(0){\mathrm{d}}\tau = \left(tI - \frac{\cos(t|{\mathbf v}|)}{|{\mathbf v}|^2} \hat{v} + \frac{1}{|{\mathbf v}|^2} \hat{v}  + \frac{2}{|{\mathbf v}|^2} \left(\frac{t}{2}-\frac{\sin(t|{\mathbf v}|)}{2|{\mathbf v}|}\right)\hat{v}^2\right) {\mathbf s}_a(0).
\end{aligned}
\end{equation}
```

Then we have 

```math
\begin{equation}
\begin{aligned}
\mathbb{M}_0 {\mathbf e}_y(t) &= \mathbb{M}_0 {\mathbf e}_y(0) - \mathbb{C}^{\mathrm{T}}\mathbb{R}^1({\mathbf X})^{\mathrm{T}}\mathbb{M}_p \int_0^{t}{\mathbf s}_{3}(\tau)d\tau,\\
\mathbb{M}_0 {\mathbf e}_z(t) &= \mathbb{M}_0 {\mathbf e}_z(0) + \mathbb{C}^{\mathrm{T}}\mathbb{R}^1({\mathbf X})^{\mathrm{T}}\mathbb{M}_p \int_0^{t}{\mathbf s}_{2}(\tau)d\tau,
\end{aligned}
\end{equation}
```

and

```math
\begin{equation}
\begin{aligned}
p_a(t) = p_a(0) - {\mathbf a}_z^{\mathrm{T}}\mathbb{C}^{\mathrm{T}}\frac{\partial R^1(x_a)}{\partial x_a}\int_0^t s_{a,2}(\tau)\mathrm{d}\tau +  {\mathbf a}_y^{\mathrm{T}}\mathbb{C}^{\mathrm{T}}\frac{\partial R^1(x_a)}{\partial x_a}\int_0^t s_{a,3}(\tau)\mathrm{d}\tau
\end{aligned}
\end{equation}
```

## Subsystem corresponding to ``H_E``

```@meta
CurrentModule = SpinGEMPIC
```
```@docs
SpinGEMPIC.operatorHE
```

```math
H_E = \frac{1}{2}{\mathbf e}_x^{\mathrm{T}}\mathbb{M}_1{\mathbf e}_x + \frac{1}{2}{\mathbf e}_y^{\mathrm{T}}\mathbb{M}_0{\mathbf e}_y + \frac{1}{2}{\mathbf e}_z^{\mathrm{T}}\mathbb{M}_0{\mathbf e}_z
```
is,

```math
\begin{equation}
\begin{aligned}
&\dot{{\mathbf X}} = {\mathbf 0},\\
&\dot{\mathbf P} = \mathbb{R}^1({\mathbf X}){\mathbf e}_x,\\
&\dot{\mathbf S} = {\mathbf 0},\\
&\dot{\mathbf e}_x = {\mathbf 0},\\
&\dot{\mathbf e}_y = {\mathbf 0},\\
&\dot{\mathbf e}_z = {\mathbf 0},\\
&\dot{\mathbf a}_y = -{\mathbf e}_y,\\
&\dot{\mathbf a}_z = -{\mathbf e}_z.\\
\end{aligned}
\end{equation}
```

We only need to solve the equation about ``{\mathbf P}``,

```math
\begin{equation}
{\mathbf P}(t) = {\mathbf P}(0) + t \mathbb{R}^1({\mathbf X}){\mathbf e}_x. 
\end{equation}
```

