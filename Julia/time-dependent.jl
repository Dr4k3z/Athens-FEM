using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Plots
using WriteVTK
using StaticArrays

function set_initial_condition_1(x)
    A = 1.
    l = 80
    #-A * sin((pi / l ) * x)
    return 0 * x - 10
end

function set_initial_speed(x)
    A = 1.
    l = 80
    return A * sin((pi / l ) * x)
end
     
function source_function(x,t)
    #X0 = 40.0
    #σ = 0.8
    #A = 8000.
    #return  - A * exp(-((x - X0)^2)/σ^2)

    t0 = 0.1
    delta = 0.001
    if t<=t0 || t>t0+delta
        return 0
    else
        return 1000000
    end
    #return 10000000
end

function matrix(n)
    #Parameter
    l = 80.#length 
    dx = l/n
    E = 1.9 * 10^7
    I = 117.8 
    # Î± = E * I / dx^4
    α = E *I / dx^4
    
    # Build the matrix
    A = spdiagm( -1 => -4 *α* ones(n-1), -2 =>  α* ones(n-2), 0 =>  6  *α* ones(n)   , 1 => - 4 *α* ones(n-1), 2 =>   α*ones(n-2) )

    # Coefficient that change in the  matrix 
    A[1,2] = -1/dx
    A[1,1]=  1/dx
    A[1,3] = 0.

    A[2,1] = 1/dx^2
    A[2,2] = -2/dx^2
    A[2,3] = 1/dx^2
    A[2,4] = 0.

    A[n-1, n-3] = 0.
    A[n-1,n-2] = 1/dx
    A[n-1,n-1] =-2/dx
    A[n-1,n] = 1/dx
    
    A[n,n-2] = 0.
    A[n,n-1] = -1/dx
    A[n,n]= 1/dx

    #time 135.690 microseconds and 65 allocations for n= 2000
    return A
end

function biharmonic!(ddu,du,u,p,t)
    E = 1.9 * 10^7
    I = 117.8
    n = 81
    A = matrix(n)
    ddu .=  A * u + (1/E*I) * source_function.(x,t)
end

N = 81
L = 80.0
dx = 1
x = Vector(0.0:dx:L)

init_t = set_initial_condition_1.(x)
init_dt = set_initial_speed.(x)   # initial speed
init_dt = ones(length(init_t))

t_begin  = 0.0
t_end = 10
tspan = (t_begin, t_end)
 
prob = SecondOrderODEProblem(biharmonic!,init_dt, init_t, tspan)

# original code by Yann - no algorithm specified - used default Tsit5()
# sol = DifferentialEquations.solve(prob)

# code modified by Domenico - specify algorithm for large stiff problems 
# https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/
sol = DifferentialEquations.solve(prob,TRBDF2())

Nt = 1000 # number of time samples 
dt = (t_end - t_begin)/Nt 

# Vector t holds 0 and is Nt+1 long 
tvec = Vector(t_begin:dt:Nt*dt)

#..interpolate solution in desired time samples 
U = zeros(2*N,length(tvec))
for k=1:length(tvec) 
  U[:,k] = sol(tvec[k])
end 

# animate
anim = @animate for i in 1:100
    plot(U[1:N,i],ylim=(0.99999,1.00000001))
end

gif(anim,fps=120)

#plot(U[1:N,1],ylim=(0.99999,1.00001))
#plot!(U[1:N,10],ylim=(0.99999,1.00001))
#plot!(U[1:N,20],ylim=(0.99999,1.00001))
#plot!(U[1:N,30],ylim=(0.99999,1.00001))
#plot!(U[1:N,40],ylim=(0.99999,1.00001))
#plot!(U[1:N,50],ylim=(0.99999,1.00001))
#plot!(U[1:N,end],ylim=(0.99999,1.00001))

plot(U[1:N,1])
plot!(U[1:N,10])
plot!(U[1:N,20])
plot!(U[1:N,30])
plot!(U[1:N,40])
plot!(U[1:N,50])
plot!(U[1:N,60])
plot!(U[1:N,70])
plot!(U[1:N,80])
plot!(U[1:N,90])
plot!(U[1:N,100])
