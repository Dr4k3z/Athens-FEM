#!/usr/bin/env julia

using LinearAlgebra
using StructArrays
using StaticArrays
using Plots
using LaTeXStrings
using SparseArrays
using BenchmarkTools 

#user defined libraries
include("mesh.jl")

function fem_1d(N)    
    if (Bool(0)) print(" [fem_1d]:: input N = ", N, "\n") end 
    
    #..Generate the mesh 
    Np1 = N+1; h = 1/N
    x = Vector(0:h:1) 
    mesh = StructArray{Element}((x[1:end-1], x[2:end], Vector(1:N), Vector(2:N+1)))

    if (Bool(0)) print(" [fem_1d]:: mesh = ", mesh, "\n") end

    #..Set the source function 
    fsource(x) = x*(x-1)

    #..Initialize global matrix and right-hand side value 
    f = zeros(Float64,Np1,1)
    I = zeros(Int64,4*N)
    J = zeros(Int64,4*N)
    Avalues = zeros(Float64,4*N)
    floc = zeros(Float64,2, 1)
    Aloc = zeros(Float64,2,2)

    #..Perform loop over elements and assemble global matrix and vector 
    @inbounds for i=1:N 
        
      xl = mesh[i].p1
      xr = mesh[i].p2
      j  = mesh[i].e1
      k  = mesh[i].e2
        
      floc = (xr-xl) * [fsource(xl), fsource(xr)];
      Aloc = (1/(xr-xl))*[1, -1, -1, 1]; 
        
      f[[j,k]] += floc 
      I[4*(i-1)+1:4*i] = [j, k, j, k]
      J[4*(i-1)+1:4*i] = [j, j, k, k]
      Avalues[4*(i-1)+1:4*i] = Aloc 
        
    end

    A = sparse(I,J,Avalues)

    #..handle the boundary conditions in the matrix and right-hand side vector 
    A[1,1] = 1;     A[1,2] = 0;        f[1]   = 0
    A[end,end-1]=0; A[end,end] = 1;    f[end] = 0

    #..solve the linear system
    u = A \ f  
    
    return x, u 
end

x,u = fem_1d(100) 

#..plot the solution  
p1 = plot(x,u,shape=:circle,lw=2,legend=false)
xlabel!("x") 
ylabel!("u(x)")
title!("Numerically computed solution")