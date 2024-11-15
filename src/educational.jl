 

function uv_mapping(azimuthal_angle,polar_angle)
    # Normalize angles to [0, 1] range

#     u = 1.0 - (azimuthal_angle) / (π)  # Azimuthal angle for centraly dominant asymmetrical background
    u = mod((azimuthal_angle) / (π),1)  # Azimuthal angle for completely symmetrical background
    v = 1-(polar_angle) / π        # Polar angle (inverted)
#     Vec2f(mod(u+0.5,1.0),v)
    return Vec2f(u,v)
end


function plot_celestial_sphere(ax,radius,rot_matrix,color_m)
    θ = LinRange(0, π, 100) .- π/2
    ϕ = LinRange(0, π, 100)
#     radius = init_r*1.1

    x = [radius * sin(ϕ) .* cos(θ') for ϕ in ϕ, θ in θ]
    y = [radius * sin(ϕ) .* sin(θ') for ϕ in ϕ, θ in θ]
    z = [radius * cos(ϕ) for ϕ in ϕ, θ in θ]

#     x .= map(x) do elem
#         if elem >-init_r + 1.5
#             return elem
#         else
#             return NaN
#         end
#     end
#     y = y[mask]
#     z = z[mask]

    points = vec([Point3f(xv, yv, zv) for (xv, yv, zv) in zip(x, y, z)])
    faces = decompose(QuadFace{GLIndex}, Tesselation(Rect(0, 0, 1, 1), size(z)))
    normals = normalize.(points)

    uv = map(zip(x,y,z)) do point
        xv, yv, zv = point
#         xv, yv, zv = rot_matrix * [point...] #(xv, yv, zv)

        return uv_mapping(atan(yv, xv),acos(zv/norm((xv,yv,zv))))
    end
#         [uv_mapping(atan(yv, xv),acos(zv/norm((xv,yv,zv)))) for (xv, yv, zv) in zip(x, y, z)]
    uv_buff = Buffer(vec(uv))

#     gb_mesh = GeometryBasics.Mesh(meta(map(x->rot_matrix*x,points); uv=uv_buff, normals), faces)
    gb_mesh = GeometryBasics.Mesh(meta(points; uv=uv_buff, normals), faces)

    surf = mesh!(ax,gb_mesh, color = color_m)
end

function plot_accretion_disk(ax=nothing)

    r = LinRange(3,6,100)
    θ = LinRange(0,2pi,100)
    x = r .* cos.(θ')
    y = r .* sin.(θ')
    z = zeros(size(y))
    if isnothing(ax)
        fig = Figure()
        ax = Axis3(fig[1,1])
        surface!(ax, x, y, z, color=fill(:white,size(z)...))# :red, highclip=:white, lowclip=:white)
        return fig
    else
        surface!(ax, x, y, z, color=fill(:white,size(z)...))#, uv_transform = compute_permutation_matrix(size(background,1),pi/4))
    end
end

