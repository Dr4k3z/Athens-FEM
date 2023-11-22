using DifferentialEquations
using Plots
using LaTeXStrings
using Symbolics

F = 1
M(x) = F*(1-x) #momentum

uEx(x) = -1/EJ .* M(x).*x.^2/2
uEx2(x) = -F/EJ * (0.5*x.^2-x.^3/6)

E(x) = x^2
Ep(x) = 2*x
Es(x) = 2
I(x) = x^2
Ip(x) = 2*x
Is(x) = 2

a(x) = (Es(x)*I(x) + 2*Ep(x)*Ip(x) + E(x)*Is(x))
b(x) = 2*(Ep(x)*I(x)+E(x)*Ip(x))
c(x) = E(x)*I(x)

function beamEqt!(du,u,p,x)
    du[1] = u[2]
    du[2] = u[3]
    du[3] = u[4]
    du[4] = (M(x) - a(x)*u[3] - b(x)*u[4])
end

function bc!(residual,u,p,x)
    residual[1] = u[1][1]
    residual[2] = u[2][1]
end

x = range(0,1,100)
xspan = (0,1)
bvp = BVProblem(beamEqt!, bc!, [0,0,0,0], xspan)
sol = solve(bvp, BS3(), dt = 0.005)

plot(sol,title="Beam deflection",xaxis="x")
