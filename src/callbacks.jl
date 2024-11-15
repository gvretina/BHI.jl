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
    if @views @inbounds get_intersection(p1,p2)
        integrator.p.flag += 1.0
        # println("White"," ", integrator.u[[2,3,4,9]])
        terminate!(integrator)
#     if @views @inbounds unew[2] > init_r
#         terminate!(integrator)
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


termination_callback() = DiscreteCallback(term_condition, term_affect!)#, ContinuousCallback(accretion_cond,affect!))
projection_callback() = DiscreteCallback(proj_condition, proj_affect!)#, ContinuousCallback(accretion_cond,affect!))
