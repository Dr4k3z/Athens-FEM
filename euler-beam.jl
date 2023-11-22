using DifferentialEquations
using Plots
using LaTeXStrings
using Symbolics

F(x) = 10*exp(-(x-0.5).^2/(2*0.1))
M(x) = F(x)*(1-x) #momentum

uEx(x) = -1/EJ .* M(x).*x.^2/2
uEx2(x) = -F/EJ * (0.5*x.^2-x.^3/6)

E(x) = x+1
I(x) = 1

function beamEqt!(du,u,p,x)
    du[1] = M(x)
    du[2] = u[1]/(E(x)*I(x))
end

function bc!(residual,u,p,x)
    residual[1] = u[1][1]
    residual[2] = u[2][1]
end

x = range(0,1,100)
xspan = (0,1)
bvp = BVProblem(beamEqt!, bc!, [0,0], xspan)
sol = solve(bvp, BS3(), dt = 0.005)

plot(sol,title="Beam deflection",xaxis="x")
