using LaTeXStrings
using RecipesBase

@userplot SpherePlot

@recipe function f(sp::SpherePlot)

    if length(sp.args) > 1
        particle_group, freq = sp.args
    else
        particle_group = first(sp.args)
        freq = 1
    end


    n = size(particle_group.array, 2)
    mask = 1:freq:n

    s1 = view(particle_group.array, 3, mask)
    s2 = view(particle_group.array, 4, mask)
    s3 = view(particle_group.array, 5, mask)
    w = view(particle_group.array, 6, mask)

    aspect_ratio --> 1
    label --> false
    seriestype --> :scatter
    markershape --> :circle
    markerstrokewidth --> 0
    markersize --> 4
    xlabel --> L"s_x" 
    ylabel --> L"s_y" 
    zlabel --> L"s_z"
    guidefontsize --> 23
    xtickfontsize --> 7
    ytickfontsize --> 7
    ztickfontsize --> 7
    legend --> false
    framestyle  -->  :box
    marker --> (1,"red")
    xlims --> (-1,1)
    ylims --> (-1,1)
    zlims --> (-1,1)

    s1, s2, s3

end


"""
    sphereplot(particle_group, freq)

It plots the sphere using spin as coordinates.
"""
sphereplot
