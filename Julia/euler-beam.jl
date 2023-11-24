using DifferentialEquations
using Plots
using LaTeXStrings
using Symbolics

F(x) = 1
M(x) = 1-x #momentum

E(x) = 1
I = 1

function beamEqt!(du,u,p,x)
    du[1] = u[2]
    du[2] = u[3]
    du[3] = u[4]
    du[4] = F(x)
end

function bc!(residual,u,p,x)
    residual[1] = u[1][1]
    residual[2] = u[end][1]
    residual[3] = u[1][3]
    residual[4] = u[end][3]
end

x = range(0,1,100)  
xspan = (0,1)
bvp = BVProblem(beamEqt!, bc!, [0.,0.,0.,0.], xspan)
sol = solve(bvp, MIRK4(), dt = 0.05)

uEx(x) = (x-2*x^3+x^4)/24

plot(sol,xaxis="x",layout=(2,2))
plot!(uEx)
