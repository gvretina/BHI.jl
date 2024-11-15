using OrdinaryDiffEq, DiffEqCallbacks
using OhMyThreads
using LinearAlgebra
using StaticArrays
using ForwardDiff
using GLMakie, GeometryBasics
# GLMakie.activate!(inline=true)
using IterTools
#import NaNMath; nm=NaNMath
using ProgressMeter
using Colors
using Images,ImageView
using Rotations
using MutableNamedTuples
const mnt = MutableNamedTuple
using LazyArrays
using LaTeXStrings
# using Interpolations
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

const D = 4
global const T = Float64
const pos_mask = [true, true, true, true, false, false, false, false]
# const tmp_mask = .~pos_mask
# tmp_mask[end] = false
const vel_mask = .~pos_mask #copy(tmp_mask)
const cart_mask = [false, true, true, true,false, false, false, false]
global const prog = Ref{Progress}(Progress(1))

const background = Images.load("a.jpg")#eso0932b
const h, w = size(background)
const init_r = Float64(50)

function kerr_metric_bl(xx::AbstractArray)

    t,r,θ,φ = xx

    M = T(1.0)
    a = T(1.0)
    @fastmath begin
    # Kerr metric components in Boyer-Lindquist coordinates

    Σ = r^2 + a^2 * cos(θ)^2
    Δ = r^2 - 2 * M * r + a^2

    # Initialize the metric tensor as a 4x4 matrix (g_tt, g_tr, g_tθ, etc.)
#     g = zeros(4, 4)

    # g_tt component (time-time)
    gtt = -(1 - 2 * M * r / Σ)

    # g_rr component (radial-radial)
    grr = Σ / Δ

    # g_θθ component (polar-polar)
    gθθ = Σ

    # g_φφ component (azimuthal-azimuthal)
    gφφ = (r^2 + a^2 + (2 * M * a^2 * r * sin(θ)^2) / Σ) * sin(θ)^2

    # g_tφ component (time-azimuthal)
    gtφ = -(2 * M * a * r * sin(θ)^2) / Σ
    end
    g = @SMatrix [gtt 0 0 gtφ;
         0 grr 0 0;
         0 0 gθθ 0;
         gtφ 0 0 gφφ]

    return g
end

function kerr_schild(xx::SVector{D,T})::SMatrix{D,D,T} where {T}
    M = 1

    t,x,y,z = xx
    #@assert !any(isnan, (t, x, y, z))
    @fastmath begin
    # <https://en.wikipedia.org/wiki/Kerr_metric>
    η = @SMatrix T[a==b ? (a==1 ? -1 : 1) : 0 for a in 1:D, b in 1:D]
    a = T(0.998)                       # T(0.8)

    ρ = sqrt(x^2 + y^2 + z^2)
    if ρ^2 - a^2<0
        r = NaN
    else
        r = sqrt( (ρ^2 - a^2)/2 + sqrt(a^2*z^2 + ((ρ^2 - a^2)/2)^2) )
    end
    f = 2*M*r^3 / (r^4 + a^2*z^2)
    k = SVector{D,T}(1,
                     (r*x + a*y) / (r^2 + a^2),
                     (r*y - a*x) / (r^2 + a^2),
                     z / r)

    g = @SMatrix T[η[a,b] + f * k[a] * k[b] for a in 1:D, b in 1:D]
    end
    g
end


function christoffel(metric::Metric, x::SVector{D,T}) where {Metric, T}
    g = metric(x)
    dg = reshape(ForwardDiff.jacobian(metric,x),(D,D,D))
    @fastmath begin

    gu = inv(g)
    Γl = @SArray T[(dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2
                   for a in 1:D, b in 1:D, c in 1:D]
    @SArray T[gu[a,1] * Γl[1,b,c] +
              gu[a,2] * Γl[2,b,c] +
              gu[a,3] * Γl[3,b,c] +
              gu[a,4] * Γl[4,b,c]
              for a in 1:D, b in 1:D, c in 1:D]
    end
end



function geodesic!(du::AbstractArray, r::AbstractArray, p, λ::T) where {T}
    x = @views @inbounds MVector{D,T}(r[pos_mask])
    u = @views @inbounds MVector{D,T}(r[vel_mask])
    Γ = christoffel(p.metric, x)

    @fastmath begin
        @inbounds for idx in 1:D
            du[idx] = r[idx+D]
            du[D+idx] = -sum(@MMatrix T[Γ[idx,b,c] * u[b] * u[c]
                                            for b in 1:D, c in 1:D])
        end
    end
        #println(du)
end

function geodesic!(du::MVector{2D,T}, r::MVector{2D,T}, p, λ::T) where {T}
    x = @views @inbounds SVector{D,T}(r[pos_mask])
    u = @views @inbounds SVector{D,T}(r[vel_mask])
    Γ = christoffel(p.metric, x)

    @fastmath begin
        @inbounds for idx in 1:D
            du[idx] = u[idx]
            du[D+idx] = -sum(@SMatrix T[Γ[idx,b,c] * u[b] * u[c]
                                            for b in 1:D, c in 1:D])
        end
    end
        #println(du)
end

hamiltonian(p,q,params) = .5*dot(p,inv(params.metric(q)),p)

function cart2polar(x::AbstractVector{T})::SVector{2,T} where T<:Real
    r = norm(x)#sqrt(x^2 + y^2 + z^2)
    θ = acos(x[3] / r)  # polar angle (0 ≤ θ ≤ π)
    φ = atan(x[2], x[1])
    SVector{2,T}(vcat(θ,φ))
end

function polar2cart(xx)
    r,θ,φ = xx
    x = r * sin(θ) * cos(φ)
    y = r * sin(θ) * sin(φ)
    z = r * cos(θ)
    @SVector [x,y,z]
end

function get_rotation_matrix(roll::Number,pitch::Number,yaw::Number)::AbstractArray

    R_euler = RotZYX(deg2rad(yaw),
                     deg2rad(pitch),
                     deg2rad(roll))

    R_euler#SMatrix{D,D,Float64,16}(ApplyArray(vcat,[0,0,0,0.]',ApplyArray(hcat,[0.,0,0],R_euler)))
end


function visible_plane(point::AbstractArray, horizontal_fov::Number, vertical_fov::Number, distance::Number)
    # Convert FOV angles from degrees to radians
    hfov_rad = deg2rad(horizontal_fov)#horizontal_fov * π / 180
    vfov_rad = deg2rad(vertical_fov)#vertical_fov * π / 180

    # Calculate half angles
    half_hfov = hfov_rad / 2
    half_vfov = vfov_rad / 2

    # Calculate the width and height of the visible plane
    width = 2 * distance * tan(half_hfov)
    height = 2 * distance * tan(half_vfov)

    # Calculate the corners of the visible plane
    x, y, z = point
    corners = [
        [sign(x)*(abs(x) - distance), y - width/2, z + height/2],  # Top-left
        [sign(x)*(abs(x) - distance), y + width/2, z + height/2],  # Top-right
        [sign(x)*(abs(x) - distance), y + width/2, z - height/2],  # Bottom-right
        [sign(x)*(abs(x) - distance), y - width/2, z - height/2]   # Bottom-left
    ]

    return reduce(hcat,corners)' |> Matrix
end

function get_visible_plane_with_angles(point::AbstractArray, h_fov::Number, v_fov::Number, dist::Number,
                                       pixels_h, pixels_w)
    # Get the angles
    angles = get_velocity_angles(v_fov, h_fov, dist, pixels_h, pixels_w)
    th, ph = angles

    # Calculate the extreme points based on angles
    corners = Vector{Float64}[]
    for theta in [th[1],th[end]]
        for phi in [ph[1],ph[end]]
            # Convert spherical coordinates to Cartesian
            x = point[1] + dist
            y = point[2] + dist * sin(phi)
            z = point[3] + dist * sin(theta)
            push!(corners, [x, y, z])
        end
    end
    return reduce(hcat,corners)' |> Matrix
end

###### weird name, just normalizes the null vector properly ######
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

function get_background_color(angs)

    θ,φ = angs

    #println(angs')
    u = 1 - mod((mod(φ,2pi)) / (π),1) # Normalize φ to range [0, 1]
    v = mod(θ,pi) / π  # Normalize θ to range [0, 1]

#     u = 1 - (φ+π) / (2π) # Normalize φ to range [0, 1]
#     v = θ / π  # Normalize θ to range [0, 1]


    # Convert UV to pixel coordinates

    px = u * w
    py = v * h

    px = round(Int,clamp(px,1,w))
    py = round(Int,clamp(py,1,h))

    # r = interp_r(py, px)
    # g = interp_g(py, px)
    # b = interp_b(py, px)

    return @inbounds background[py,px]#RGB{N0f8}(r, g, b)
end

function get_intersection(p1,p2)

    x1, y1, z1 = p1 #previous
    x2, y2, z2 = p2 #current

    if (abs(z2) < 5e-3) && (3 < norm(p2) < 6)
        return true
    elseif z1 * z2 > 0
        return false  # Both points are above or below the disk
    elseif z2 != z1
        t = -z1 / (z2 - z1)
        x_intersect = x1 + t * (x2 - x1)
        y_intersect = y1 + t * (y2 - y1)

        # Check if the intersection point is within the disk's radius
        distance_to_center =  norm((x_intersect, y_intersect))#sqrt(x_intersect^2 + y_intersect^2)
        return 3 < distance_to_center < 6
    else
        return false
    end

end

term_condition(u, λ, integrator) = true
# begin
#     r = @views @inbounds norm(SVector{3,T}(u[cart_mask]))
#      ((r > (1.1*init_r)) ### goes to infty
#         ||
#             (r < 1.5) ### hits BH horizon r ~ 1.435889
#         ||  (@views @inbounds get_intersection(integrator.uprev[cart_mask],integrator.u[cart_mask]) )
#         )
# end

function term_affect!(integrator)
#     integrator.saveiter = 0
#     r = @views @inbounds norm(SVector{3,T}(integrator.u[cart_mask]))
    r = @inbounds integrator.u[2]
#     println(integrator.u)
#     global wtf = integrator
    if r < 1.05
        integrator.p.flag += -1.0
        terminate!(integrator)
#     elseif (@views @inbounds get_intersection(polar2cart(integrator.uprev[cart_mask]),polar2cart(integrator.u[cart_mask])) )
#         integrator.p.flag += 1.0
#         println("yes")
#         terminate!(integrator)
    elseif r > 1.001*init_r
        terminate!(integrator)
    end
end

proj_condition(u, λ, integrator) = true
############### write these properly ##############
function proj_affect!(integrator)
#     global wtf = integrator
#     throw("No")
    u = integrator.u
    dt = integrator.dt
    p = integrator.p
    g = constraints
    g_prime = ForwardDiff.jacobian(x->g(x,p),u)
    inv_g_sq = inv(g_prime*g_prime')
#     display(inv_g_sq)
#     println(g(u,p))
    n = size(g(u,p),1)
    λ = zeros(n) ### Lagrange multipliers
    Δλ = zeros(n) ### Lagrange multipliers
    i = 1
    tol = 100*dt^(p.order+1)
    prev = zeros(n)

    while true
#         g_prime = ForwardDiff.jacobian(x->g(x,p),u)
#         inv_g_sq = inv(g_prime*g_prime')

        if mod(i,50) == 0
            g_prime .= ForwardDiff.jacobian(x->g(x,p),u)
            inv_g_sq .= inv(g_prime*g_prime')
        end


        Δλ = -inv_g_sq * g(u+g_prime'*λ,p)

        λ += Δλ
        val = norm(Δλ)
#         if i > 100
#             @views @inbounds println(val," ", tol,"       ",integrator.u[cart_mask])
#             println(norm(prev-Δλ))
#             # if i == 200
#                 # throw("Stop")
#             # end
#         end
        if any(isnan.(Δλ))
            global wtf = integrator
            throw("Wrong")
        end
        # throw("No")
        if val < max(tol,2e-16) || i > 1000
            if i > 1000
                println(val)
            end
            # println("Next")
            break
        end
        i += 1
        prev .= Δλ
    end
    unew =  u .+ g_prime'*λ

#     if mod(integrator.saveiter,2) == 0
#         integrator.saveiter += 1
#     end

    copyto!(integrator.u, unew)
    integrator.u_modified = false
    integrator.saveiter -= 1
    if p.save == false
        integrator.saveiter = 0
    end
    p1 = @views @inbounds polar2cart(integrator.uprev[cart_mask])
    p2 = @views @inbounds polar2cart(integrator.u[cart_mask])
    dist = norm(p1-p2)
#     if @inbounds integrator.uprev[2] < 5 && dist > 1.5
#         println(p1)
#         println(p2)
#         println()
#     end
#
#     u = integrator.u
#     if @views @inbounds get_intersection(p1,p2)
#         integrator.p.flag += 1.0
#         # println("White"," ", integrator.u[[2,3,4,9]])
#         terminate!(integrator)
    if @views @inbounds unew[2] > init_r
        terminate!(integrator)
    elseif @views @inbounds unew[2] < 1.15 || (dist > 5 && integrator.uprev[2] < 2)
        integrator.p.flag += -1.0
        # println("Black"," ", integrator.u[[2,3,4,9]])
        terminate!(integrator)
    end    

end

function constraints(u,p)
    g = @views @inbounds p.metric(u[pos_mask])
    lagrange = @views @inbounds dot(u[vel_mask],g,u[vel_mask])

    gtt = g[1,1]
    gtφ = g[1,4]
    gφφ = g[4,4]
    energy = gtt*u[D+1]+gtφ*u[2D] - p.E
    angular_momentum = gφφ*u[2D]+gtφ*u[D+1] - p.L
    return [lagrange, energy, angular_momentum]
end


function constr(resid, u, p, t)
    g = @views @inbounds p.metric(u[pos_mask])
    gtt = g[1,1]
    gtφ = g[1,4]
    gφφ = g[4,4]

    lagrange = @views @inbounds dot(u[vel_mask],g,u[vel_mask])
    energy = @inbounds gtt*u[D+1]+gtφ*u[2D] - p.E
    angular_momentum = @inbounds gφφ*u[2D]+gtφ*u[D+1] - p.L

    resid[1] = lagrange#u[2]^2 + u[1]^2 - 2
    resid[2] = energy
    resid[3] = angular_momentum
end

const cb1 = DiscreteCallback(term_condition, term_affect!)#, ContinuousCallback(accretion_cond,affect!))
const cb2 = DiscreteCallback(proj_condition, proj_affect!)#, ContinuousCallback(accretion_cond,affect!))
const cb = CallbackSet(ManifoldProjection(constr),cb1)
############### END ##############

function integrate_geodesics_edu(input,metric)
    tspan = (T(0),T(3000))
    tol=1e-12#eps(T)^(3/4)

    gtt = @views @inbounds metric(input[pos_mask])[1,1]
    gtφ = @views @inbounds metric(input[pos_mask])[1,4]
    gφφ = @views @inbounds metric(input[pos_mask])[4,4]
    E = gtt*input[D+1]+gtφ*input[2D]#-sum([g[a,b]*u[b] for a)
    L = gφφ*input[2D]+gtφ*input[D+1]
    p = mnt(metric=metric, order = 7, flag = 0.0, save=true, E=E, L=L)

    prob = ODEProblem(geodesic!, input, tspan, p)

    (@isdefined cbb) ? cbb=CallbackSet(cb,cbb) : cbb=cb

    sims = solve(prob, Vern7(), callback=cb2, #dt = 1e-5, adaptive = false, progress = true, save_start=false, save_end = false)
                 abstol=tol, reltol=tol, dense=false, progress = true)
                 #dtmin=1e-3, force_dtmin=true, calck=false)
    sol = @views @inbounds reduce(hcat,sims.u)'#[:,[2,3,4]]
#     println(sims.t[end])
    sims = nothing
    return sol
end


function integrate_geodesics_test(input,metric)
    tspan = (T(0),T(3500.0))
    tol=1e-15#eps(T)^(3/4)

    gtt = @views @inbounds metric(input[pos_mask])[1,1]
    gtφ = @views @inbounds metric(input[pos_mask])[1,4]
    gφφ = @views @inbounds metric(input[pos_mask])[4,4]
    E = gtt*input[D+1]+gtφ*input[2D]#-sum([g[a,b]*u[b] for a)
    L = gφφ*input[2D]+gtφ*input[D+1]
    p = mnt(metric=metric, order = 5, flag = 0.0, save=true, E=E, L=L)

    prob = ODEProblem(geodesic!, input, tspan, p)

    (@isdefined cbb) ? cbb=CallbackSet(cb,cbb) : cbb=cb

    dt = 1/1024#tspan[end]/4e5
    max_iter = round(Int64,tspan[end]/dt)+1
    sims = solve(prob, Tsit5(), callback=cb2, dt = dt, adaptive = false, progress = true, maxiters=max_iter)#, save_start=false, save_end = false)
#                  abstol=tol, reltol=tol, dense=false)#, progress = true)
                 #dtmin=1e-3, force_dtmin=true, calck=false)
    sol = @views @inbounds reduce(hcat,sims.u)'#[:,[2,3,4]]
    println("T = ",sims.t[end])
#     println("N steps = ", length(sims.t))
    t = sims.t
    sims = nothing
    return sol,t
end

function integrate_geodesics_scientific(input,metric)
    tspan = (T(0),T(3000))
    tol=1e-10#eps(T)^(3/4)

    gtt = @views @inbounds metric(input[pos_mask])[1,1]
    gtφ = @views @inbounds metric(input[pos_mask])[1,4]
    gφφ = @views @inbounds metric(input[pos_mask])[4,4]
    E = gtt*input[D+1]+gtφ*input[2D]#-sum([g[a,b]*u[b] for a)
    L = gφφ*input[2D]+gtφ*input[D+1]
    p = mnt(metric=metric, order = 7, flag = 0.0, save=false, E=E, L=L)
    prob = ODEProblem(geodesic!, input, tspan, p)
#     prob = HamiltonianProblem(hamiltonian, input[1], input[2], tspan, p)

    (@isdefined cbb) ? cbb=CallbackSet(cb,cbb) : cbb=cb
    
    #     sims = solve(prob,KahanLi8(),# dt=1e-2, callback=cb1)
    sims = solve(prob, Vern7(), callback=cb2,
     abstol=tol, reltol=tol, dense=false,
    #             #  dt = 1e-3, adaptive = false,
    #              save_start=false,save_everystep=false,
    #              save_end=true, calck=false, 
                 dtmin=1e-3, force_dtmin=true)#,save_idxs=[2,3,4,9])#.u[end]
    sol = @views @inbounds (sims.u[end][cart_mask], sims.prob.p.flag)#[end,:]
    # println(length(sims.u))
    # throw("No")
    
    sims = nothing
    return sol
end

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


function trace_rays_edu(metric::Metric, sc::Int64) where Metric

    ### Initial Conditions ###
    dist = T(1)

    ####### Velocities ######## TODO make matrix with all combinations and feed that to updater
    #h,w = size(background)
    scaler = h>sc ? h÷sc : -1
    if scaler != -1
        N,M = h÷scaler,w÷scaler
    else
        N,M = sc,sc
    end

    vfov = 2#60#hfov * N ÷ M
    hfov = 2#60#(vfov*M)÷N
    hfov > 90 ? hfov = 90 : nothing
    M = N#div(N,10)
    println(N," ",M)

    alims = (-10,10) ### pθ/pt impact param
    blims = (-10,10) ### -pφ/pt impact param

    roll = 0
    pitch = 85
    yaw = 0
    θ0 = deg2rad(pitch)

    R_euler = get_rotation_matrix(roll,pitch,yaw)
    pos = SVector{D,T}([0,init_r,θ0,π])
    inputs = generate_rays(pos, metric, N, M, alims, blims, dist)#zeros(MVector{2D + 1,T}, N, M)

#     println(pos)
#     println(inputs[1])
#     throw("No")
    prog[] = Progress(N*M)
#     p = (metric=metric, prog=prog, flag=0, order=7)

    sols = tmap(1:(N*M)) do i
        sol = integrate_geodesics_edu(inputs[i],metric)
        next!(prog[])
        if mod(i,div(N*M,Threads.nthreads())) == 0
            GC.gc()
        end
        return sol
    end
    GC.gc()

    fig = Figure()
    ax = LScene(fig[1,1],show_axis=false,
                scenekw = (
                    lights = [DirectionalLight(
                        RGB{N0f8}(0.3,0.3,0.3),#colorant"white",
                        Point3f(R_euler*[-init_r-0.5,0,0])),
                    AmbientLight(colorant"white")],
                    ))#Axis3(fig[1,1],xlabel="x",ylabel="y",zlabel="z")
    mesh!(ax,Sphere(Point3f(0), 1.5f0), color = :black)
    color_m = Sampler(background)#, x_repeat=:repeat,y_repeat=:repeat) # circshift(background,(0,div(w,2))) for horizontal shift
    plot_accretion_disk(ax)
    mesh!(ax,Sphere(Point3f(polar2cart(pos[2:D])), .25f0), color = :purple)

    corners =  visible_plane(polar2cart(pos[2:end]),hfov,vfov,dist*4)

    #global wtf = reshape(sols,N,M)

    back = circshift(background,(0,div(w,2)))
    img = map(reshape(sols,N,M)) do sims
        pos = @views sims[end,:]#.u[end]#[cart_mask]
        fin_r = pos[1]#norm(pos)
        angs = pos[2:3]#cart2polar(pos)

         if get_intersection(polar2cart(sims[end-2,:]),polar2cart(pos))#3 < fin_r < 6 && abs(angs[1] - pi/2) < 5e-3
             return RGB{N0f8}(1.0, 1.0, 1.0)
        elseif fin_r < 1.5
             return RGB{N0f8}(0.0, 0.0, 0.0)
         else
             return get_background_color(angs)#background[get_background_pixel_idx(angs)]
         end
    end

    n_traj = N*M
    n_shown = 100#0
    step = n_traj ÷ n_shown
    n_traj <= 100 ? indices = (1:n_traj) : indices = rand(1:n_traj,n_shown)#1:step:n_traj

    rmax = 0

    @showprogress for i in indices
        sol = reduce(hcat,map(polar2cart,eachrow(sols[i])))'
#         display(sol)
        lines!(ax,sol[:,1],sol[:,2],sol[:,3], color=img[i])
        r = @views @inbounds norm(sol[end,:])
        if r > rmax
            rmax = r
        end
    end

    plot_celestial_sphere(ax,rmax,R_euler,color_m)

    sols = nothing
    GC.gc()

    x = corners[1,1]
    ymin,ymax = minimum(corners[:,2]),maximum(corners[:,2])
    zmin,zmax = minimum(corners[:,3]),maximum(corners[:,3])
    y = LinRange(ymin,ymax,size(img,1))
    z = LinRange(zmin,zmax,size(img,2))

    imm = image(fig[1,2],rotl90(img)',#rot180(reverse(img,dims=1)),
                axis = (aspect = DataAspect(),))
    hidespines!(imm.axis)
    hidedecorations!(imm.axis)
    display(fig)
#     update_cam!(ax.scene, (R_euler * @SVector([-init_r,0,0])), (R_euler*[1, 0, 0]), (R_euler*[0, 0, 1]))
    fig
end

function trace_rays_scientific(metric::Metric, sc::Int64) where Metric

    ### Initial Conditions ###
    dist = T(1)

    ####### Velocities ######## TODO make matrix with all combinations and feed that to updater
    #h,w = size(background)
    scaler = h>sc ? h÷sc : -1
    if scaler != -1
        N,M = h÷scaler,w÷scaler
    else
        N,M = sc,sc
    end
#     vfov = 2#60#hfov * N ÷ M
#     hfov = 2#60#(vfov*M)÷N
#     hfov > 90 ? hfov = 90 : nothing
    M = N
    println(N," ",M)
#     println(vfov," ",hfov)

#     roll = 0
#     pitch = 10
#     yaw = 0

    θ0 = deg2rad(87.5)

    alims = (-9,9) ### pθ/pt impact param
    blims = (-9,9) ### -pφ/pt impact param

    #R_euler = get_rotation_matrix(roll,pitch,yaw)
    pos = SVector{D,T}([0,init_r,θ0,π/2])
    inputs = generate_rays(pos, metric, N, M, alims, blims, dist)#zeros(MVector{2D + 1,T}, N, M)

    prog[] = Progress(N*M)

    sols = map(1:(N*M)) do i
        sol = integrate_geodesics_scientific(inputs[i],metric)
        next!(prog[])
        # GC.gc()
        # if mod(i,1000) == 0#div(N*M,Threads.nthreads())) == 0
            # GC.gc()
        # end
        return sol
    end


    # GC.gc()
    # global wtf = copy(sols)
    # println(sols)
    img = map(reshape(sols,N,M)) do sims
        pos = @views @inbounds sims[1]
        #println(sims)
        # println(sims)
        fin_r = pos[1]#norm(pos)
        angs = @views @inbounds pos[2:3]#cart2polar(pos)

        if sims[end] >  0.0#3 < fin_r < 6 && abs(angs[1] - pi/2) < 5e-3
             return RGB{N0f8}(1.0, 1.0, 1.0)
        elseif sims[end] < 0.0
             return RGB{N0f8}(0.0, 0.0, 0.0)
        else
             return get_background_color(angs)#background[get_background_pixel_idx(angs)]
        end
    end

#     fig = Figure()
#     imm = image(fig[1,1],rotl90(img),
#              axis = (aspect = DataAspect(),))
#     hidespines!(imm.axis)
#     hidedecorations!(imm.axis)
    # display(fig)
    # update_cam!(ax.scene, (R_euler * @SVector([-init_r,0,0])), (R_euler*[1, 0, 0]), (R_euler*[0, 0, 1]))
#     save("trace.png",fig)
    img .= reverse(imrotate(img,-π/2),dims=1)
    imshow(img)
    save("trace_rad.png",img)
#     fig
end

function trace_rays(metric,sc,flag="sci")

    if flag == "edu"
        trace_rays_edu(metric,sc)
    elseif flag == "sci"
        trace_rays_scientific(metric,sc)
    else
        throw("Not supported method")
    end
end

function energy(metric,x)
    pos = @views @inbounds x[pos_mask]
    vel = @views @inbounds x[vel_mask]
    g = metric(pos)
    gtt = @views @inbounds g[1,1]
    gtφ = @views @inbounds g[1,4]
    gφφ = @views @inbounds g[4,4]

    E = @inbounds gtt*vel[1]+gtφ*vel[end]#-sum([g[a,b]*u[b] for a)
    E
end

function ang_momentum(metric,x)
    pos = @views @inbounds x[pos_mask]
    vel = @views @inbounds x[vel_mask]
    g = metric(pos)
    gtt = @views @inbounds g[1,1]
    gtφ = @views @inbounds g[1,4]
    gφφ = @views @inbounds g[4,4]

    L = @inbounds gφφ*vel[end]+gtφ*vel[1]
    L
end

function lagrange(metric,x)
    pos = @views @inbounds x[pos_mask]
    vel = @views @inbounds x[vel_mask]
    g = metric(pos)
    vel' * g * vel
end

function robustness()
    metric = kerr_metric_bl
    pos = @SVector [0.0,50.0,π/2,1.0]
#     pos = @SVector [0.0,1.0+sqrt(3.0),π/2,0.0]
    g = metric(pos)
    gu = inv(g)

    rd = gu*[0.0,-1.0,0.0,-2.0972915947602124]#-2.0951938862158537]
    rd = SVector{D,T}(vcat(0, rd[[2,3,4]]))
    ray_direction = get_final_velocity(rd,g,gu)
    println(ray_direction' * g * ray_direction)
#     ray_direction = gu*[-1.0,0.0,sqrt(12.0+8.0*sqrt(3.0)),-1.0]
#     ray_direction =  SVector{D,T}(vcat(0, ray_direction[[2,3,4]]))
    ray = MVector{2D,T}(vcat(pos,ray_direction))

#     println(ray[D+1:end]' * g * ray[D+1:end])

#         LScene(fig[1,1],show_axis=false,
#                 scenekw = (
#                     lights = [
#                     AmbientLight(colorant"white")],
#                     ))#Axis3(fig[1,1],xlabel="x",ylabel="y",zlabel="z")
#     mesh!(ax,Sphere(Point3f(0), 1.05f0), color = :black)

    sol,λ = integrate_geodesics_test(ray,metric)
    pl = reduce(hcat,map(eachrow(sol)) do x
                    @views @inbounds polar2cart(x[cart_mask])
    end
                    )'
    N = size(sol,1)
    pow = round(Int,log10(N))

    set_theme!(theme_latexfonts())
    fig = Figure(fontsize=26,size=(1920,1080),pt_per_unit=1,px_per_unit=1)
    ax1 = Axis3(fig[1,3])
    mesh!(ax1,Sphere(Point3f(0), 1.15f0), color = :black)
    xlims!(ax1,-5,5)
    ylims!(ax1,-5,5)
    zlims!(ax1,-5,5)

    stepstring = latexstring(L"{\mathrm{steps}} \cdot 10^{"[1:end-1] * string(pow) * "}\$")
    ax6 = GLMakie.Axis(fig[2,3],xlabel="λ",ylabel="r")

    ax2 = GLMakie.Axis(fig[1,1],ylabel=L"\Delta r/r_0", xlabel=stepstring, yscale=log10)
    ax22 = GLMakie.Axis(fig[1,1], xlabel="λ", yscale=log10, xaxisposition=:top)
    hidespines!(ax22)
    hideydecorations!(ax22)


    ax3 = GLMakie.Axis(fig[1,2],ylabel=L"\Delta e/e_0",xlabel=stepstring, yscale=log10)
    ax4 = GLMakie.Axis(fig[2,1],ylabel=L"\Delta l/l_0",xlabel=stepstring, yscale=log10)
    ax5 = GLMakie.Axis(fig[2,2],ylabel=L"g_{\mu \nu} u^\mu u^\nu",xlabel=stepstring)

    r = sol[1,2]
    Δr = @views @inbounds abs.(sol[2:end,2].-r)#./r
    mask = Δr .== 0.0
    Δr[mask] .= 1e-16
    nsteps = 1:N

    Δe = map(eachrow(sol)) do x
             energy(metric,x)
        end
    Δl = map(eachrow(sol)) do x
        ang_momentum(metric,x)
    end
    Δn = map(eachrow(sol)) do x
        lagrange(metric,x)
    end
    Δe = abs.(diff(Δe)./Δe[1])
    mask = Δe .== 0.0
    Δe[mask] .= 1e-16
    Δl = abs.(diff(Δl)./Δl[1])
    mask = Δl .== 0.0
    Δl[mask] .= 1e-16
    lines!(ax1,pl[1:2:end,1],pl[1:2:end,2],pl[1:2:end,3])
    scatterlines!(ax6,λ,sol[:,2])

    lines!(ax2,nsteps[1:end-1]./(10^pow),Δr)
    p = lines!(ax22,λ[1:end-1],Δr, alpha=0.001)
    display(fig)
    delete!(ax22,p)
    lines!(ax3,nsteps[1:end-1]./(10^pow),Δe)
    lines!(ax4,nsteps[1:end-1]./(10^pow),Δl)
    lines!(ax5,nsteps./(10^pow),Δn)
    save("errors.png",fig)
end

function find_ic(x)
    metric = kerr_metric_bl
    pos = @SVector [0.0,50.0,π/2,0.0]
#     pos = @SVector [0.0,1.0+sqrt(3.0),π/2,0.0]
    g = metric(pos)
    gu = inv(g)
    rd = gu*[0.0,-1.0,0.0,x]#-2.0951938862158537]
    rd = SVector{D,T}(vcat(0, rd[[2,3,4]]))
    ray_direction = get_final_velocity(rd,g,gu)

#     ray_direction = gu*[-1.0,0.0,sqrt(12.0+8.0*sqrt(3.0)),-1.0]
#     ray_direction =  SVector{D,T}(vcat(0, ray_direction[[2,3,4]]))
    ray = MVector{2D,T}(vcat(pos,ray_direction))

    sol,λ = integrate_geodesics_test(ray,metric)
    return λ[end]
end

# Function to evaluate fitness in parallel using threads
function evaluate_fitness_threads(black_box_function, population)
    fitness = Vector{Float64}(undef, length(population))
    @threads for i in 1:length(population)
        fitness[i] = black_box_function(population[i])
    end
    return fitness
end

# Genetic Algorithm with threaded fitness evaluation
function genetic_algorithm_adaptive_threads(black_box_function, population_size, generations, mutation_rate, initial_search_range)
    search_range = initial_search_range
    population = [rand(search_range) for _ in 1:population_size]

    # Parallel fitness evaluation using threads
    fitness = evaluate_fitness_threads(black_box_function, population)

    for gen in 1:generations
        # Selection: choose the top 50% as parents
        sorted_indices = sortperm(fitness, rev=true)
        parents = population[sorted_indices[1:div(population_size, 2)]]

        # Crossover: generate new candidates by averaging parents
        offspring = []
        for _ in 1:div(population_size, 2)
            p1, p2 = rand(parents, 2)
            child = (p1 + p2) / 2
            push!(offspring, child)
        end

        # Mutation: apply small random changes to offspring
        for i in 1:length(offspring)
            if rand() < mutation_rate
                offspring[i] += (rand() - 0.5) * (search_range[2] - search_range[1]) * 0.1
            end
        end

        # Combine parents and offspring to form the new population
        population = vcat(parents, offspring)

        # Evaluate the new fitness in parallel using threads
        fitness = evaluate_fitness_threads(black_box_function, population)

        # Print the best fitness of each generation
        println("Generation $gen, Best fitness: $(maximum(fitness))")

        # Update the search range to focus on the best candidates
        top_individuals = population[sorted_indices[1:div(population_size, 5)]]
        new_center = mean(top_individuals)
        new_range_size = (search_range[2] - search_range[1]) * 0.9  # Gradually reduce range size

        # Ensure the search range doesn't shrink too fast
        search_range = (new_center - new_range_size / 2, new_center + new_range_size / 2)
    end

    # Return the best solution found
    best_index = argmax(fitness)
    return population[best_index], fitness[best_index]
end


println("Starting...")
# trace_rays(kerr_schild,100)
#fig
