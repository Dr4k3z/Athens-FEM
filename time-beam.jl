using DifferentialEquations

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
     
function source_function(x)
    X0 = 40.0
    σ = 0.8
    A = 80.
    return  - A * exp(-((x - X0)^2)/σ^2)
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
    ddu .=  A * u + (1/E*I) * source_function.(x)
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

Nt = 10 # number of time samples 
dt = (t_end - t_begin)/Nt 

# Vector t holds 0 and is Nt+1 long 
tvec = Vector(t_begin:dt:Nt*dt)

#..interpolate solution in desired time samples 
U = zeros(2*N,length(tvec))
for k=1:length(tvec) 
  U[:,k] = sol(tvec[k])
end 

p1 = plot(U[1:N,1], label = "0s")
plot!(U[1:N,2], label = "1s")
plot!(U[1:N,3], label = "2s")
plot!(U[1:N,4], label = "3s")
xlabel!("x (m)")
ylabel!("position(m)")
title!("Downward deflection")

p2 = plot(U[1:N,5], label = "4s")
plot!(U[1:N,6], label = "5s")
plot!(U[1:N,7], label = "6s")
plot!(U[1:N,8], label = "7s")
xlabel!("x (m)")
ylabel!("position(m)")
title!("Oscillation of the deflection")

p3 = plot(U[1:N,9], label = "8s")
plot!(U[1:N,10], label = "9s")
plot!(U[1:N,11], label = "10s")
xlabel!("x (m)")
ylabel!("position(m)")
title!("Return of the deflection")


plot(p1,p2,p3, size=(1500,1000), layout=(1,3))