using Plots, Sobol

n = 10000
alpha, kx = 0.1, 0.5
xmin, xmax = 0.0, 2π/kx
dx = (xmax - xmin) / (n-1)
x = LinRange(xmin, xmax, n) |> collect
f = 1 .+ alpha*cos.(kx .* x)
dy = 2.0 / (n-1)
y = LinRange(-1, 1, n) |> collect
g = ( 1 .+ y ./ 2)
v = cumsum(f)*dx 
w = cumsum(g)*dy 
s  = SobolSeq(1)
xp = Float64[]
sp = Float64[]
for k=1:n
   r = next!(s)[1]
   i = findmin(abs.(v .- r * 4π) )[2]
   j = findmin(abs.(w .- 2r))[2]
   push!(xp,  x[i])
   push!(sp,  y[j])
end

p = plot(layout=(2,1))
histogram!(p[1], xp, normalize=true, bins = 100)
plot!(x-> (1+alpha*cos(kx*x))/4π, 0., 4π)
histogram!(p[2], sp, normalize=true, bins = 100)
plot!(p[2], x -> (1 + x / 2) / 2, -1, 1, lab="")
