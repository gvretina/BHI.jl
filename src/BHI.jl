module BHI

using ProgressMeter
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using Images,ImageView
using LaTeXStrings

const D = 4
global const T = Float64
const pos_mask = [true, true, true, true, false, false, false, false]
# const tmp_mask = .~pos_mask
# tmp_mask[end] = false
const vel_mask = .~pos_mask #copy(tmp_mask)
const cart_mask = [false, true, true, true,false, false, false, false]
global const prog = Ref{Progress}(Progress(1))

include("callbacks.jl")
include("initial_conditions.jl")
include("educational.jl")
include("metrics.jl")
include("integrator.jl")

const background = Images.load("src/background_images/a.jpg")#eso0932b
const h, w = size(background)
const init_r = T(500.0)


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

function get_background_color(angs,background=nothing)

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
             return get_background_color(angs,background)#background[get_background_pixel_idx(angs)]
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
             return get_background_color(angs,background)#background[get_background_pixel_idx(angs)]
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


end
