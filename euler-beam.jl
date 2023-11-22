#!/usr/bin/env julia

using DifferentialEquations
using Plots

F = 1
M(x) = F*(1-x) #momentum
EJ = 1
function beamEqt!(du,u,p,x)
    du[1] = u[2]
    du[2] = -M(x)
end

function bc!(residual,u,p,x)
    residual[1] = u[1][1]
    residual[2] = u[2][1]
end

h = 0.01
xspan = (0,1)
bvp1 = BVProblem(beamEqt!, bc!, [0,0], xspan)
sol1 = solve(bvp1, BS3(), dt = 0.005)
plot(sol1)

uEx(x) = -1/EJ .* M(x).*x.^2/2
uEx2(x) = -F/EJ * (0.5*x.^2-x.^3/6)
plot!(uEx2)

