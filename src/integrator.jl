using OrdinaryDiffEq, DiffEqCallbacks
using ForwardDiff
using MutableNamedTuples
const mnt = MutableNamedTuple

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
    x = @views @inbounds SVector{D,T}(r[pos_mask])
    u = @views @inbounds SVector{D,T}(r[vel_mask])
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

    sims = solve(prob, Vern7(), callback=cb2(), #dt = 1e-5, adaptive = false, progress = true, save_start=false, save_end = false)
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

    #     sims = solve(prob,KahanLi8(),# dt=1e-2, callback=cb1)
    sims = solve(prob, Vern7(), callback=cb2(),
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
