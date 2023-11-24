using Plots
using SparseArrays


# plot to see intial_function
function init()
    F = Real[]
    space = Real[]
    for x in 0:0.5:80
        f(x) = 100 * sin((pi / 80)*x)
        push!(F, f(x))
        push!(space, x)
    end
    plot(space,F)
end 

#-------------------------------------------------------------------------------------------------------------------------------------

function exact_solution(n)
#Parameter for copper
    i = 5.0
    d= 8.92 #density 
    sh= 0.092 #specific_heat
    tc= 0.95 #thermal_conductivity
    l= 80 #length
    c = ( tc / (sh * d))^(1/2)
    t = 5.0#fix the time variable 
    lambda_i = (i * c * pi) / l
    dx = l/ (n -1) #n is the number of point with the starting point 
    u_exact = zeros(n)
    x = 0.0
        

 #Define exact solution 
    for i in 1:n
        u_exact[i] = 100 * sin(pi * x /80) * exp(-lambda_i^2 * t)
        x = x + dx
    end
    #R = [k for k in 0:dx:l]      
    #plot(R,u_exact)
    return u_exact
end

#------------------------------------------------------------------------------------------------------------------------------------

function heat(n)
    #Parameter for copper
    d= 8.92 #density 
    sh= 0.092 #specific_heat
    tc= 0.95 #thermal_conductivity
    l= 80.0 #length
    c = ( tc / (sh * d))^(1/2)
    dx = l/ (n -1) #n is the number of point with the starting point 
    tmax = 30.0 #30 secondes 
    dt = 0.001
    n_iteration = Int(tmax /dt) + 1 #to convert into integer
    α = dt * c^2 /dx^2 #Without dt for now 
    #α =  dt*c^2 /dx^2
    
    #Boundary solution left = 0 and right = 0
    U = zeros(n)
    U[1] = 0.0
    U[n] = 0.0  

    #inital condition for Boundary left = 0 and right = 0 so it's u_exact for t = 0
    for i in 2:n-1
        x = (i-1)*dx
        U[i] = 100* sin((pi / l)*x)
    end 

    Unp1 = copy(U)
    #Solve the system
    for j in 1:n_iteration
        for i in 2:n-1
            Unp1[i] = α * (U[i+1] - 2 * U[i] + U[i-1]) + U[i]
        end 
        U = Unp1
    end 
    return U
end


#plot the error of the function 
function error(n)

    l = 80.0
    dx = l/ (n -1)
    exact = exact_solution(n)
    modele = heat(n)
    erreur = zeros(n)

    for i in 1:n
        erreur[i] = abs(exact[i] - modele[i])
    end
    X = [i for i in 0:dx:l]
    plot(X, erreur)
end


#plot the exact_solution and the heat equation solving 
function curbes(n)
    l = 80.0
    dx = l/ (n -1)
    exact = exact_solution(n)
    modele = heat(n)
    X = [i for i in 0:dx:l]
    plot(X, [exact,modele])
end 


