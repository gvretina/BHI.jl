using LinearAlgebra

function normalization(normal::AbstractArray,#Vector{D,T},
                            g::Metric, gu::Metric) where Metric

    @assert normal[1] == zero(eltype(normal))

    @fastmath begin
#     normal = SVector{D,T}(normal./norm(normal))
    t = gu * SVector{D,T}(1, 0, 0, 0)
    t2 = t' * g * t
    n2 = normal' * g * normal
    normal = SVector{D,T}((t / sqrt(-t2) + normal / sqrt(n2)))
    end
    normal
end

###### preps initial data #####
function generate_rays(pos, metric, height, width, alims,blims, dist)
    # Create an array to store ray directions for each pixel
    rays = Array{MVector{2D,Float64}}(undef, height, width)
    # FOV adjustments for horizontal and vertical directions
    g = metric(pos)
    gu = inv(g)

    as = LinRange(alims[1],alims[2],width)
    bs = LinRange(blims[1],blims[2],height)

    for (j,a) in enumerate(as)
        for (i,b) in enumerate(bs)
            ### do this properly with normalizations and stuff
            pt = 1.0 # = -E
            a += T(1e-12)
            b += T(1e-12)

            ray_direction = gu*[0.0, -1.0, a*pt, b*pt]
            ray_direction =  SVector{D,T}(vcat(0, ray_direction[[2,3,4]]))
#             println(ray_direction)
            rays[i, j] = vcat(pos,normalization(ray_direction,g,gu))
#             println(rays[i,j][D+1:end]' * g * rays[i,j][D+1:end])
#             throw("Stop")
        end
    end

    return rays
end
