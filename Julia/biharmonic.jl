using LinearAlgebra
using StructArrays
using StaticArrays
using Plots
using LaTeXStrings
using SparseArrays

include("mesh.jl")

fsource(x) = 1

N = 16
h = 1/N
xnod = range(0,1,N+1)
#..Generate the mesh 
x = Vector(0:h:1) 
mesh = StructArray{Element}((x[1:end-1], x[2:end], Vector(1:N), Vector(2:N+1)))

I = zeros(Int64,4*N)
J = zeros(Int64,4*N)
Avalues = zeros(Float64,4*N)
f = zeros(Float64,(N+1),1)

floc = zeros(Float64,2, 1)
Aloc = zeros(Float64,2,2)

@inbounds for i=1:N
    xl = mesh[i].p1; xr = mesh[i].p2
    j = mesh[i].e1; k = mesh[i].e2;
    floc = (xr-xl)*[fsource(xl), fsource(xr)];
    Aloc = (1/(xr-xl))*[1, -1, -1, 1]; 

    f[[j,k]] = floc
    I[4*(i-1)+1:4*i] = [j, k, j, k]
    J[4*(i-1)+1:4*i] = [j, j, k, k]
    Avalues[4*(i-1)+1:4*i] = Aloc
end

A = sparse(I,J,Avalues)
#A[1,1] = 1; A[1,2] = 0; A[2,1] = 0
#A[end,end]=1; A[end,end-1] = 0; A[end-1,end] =0;

A[1,1] = 1;     A[1,2] = 0;        f[1]   = 0 
A[end,end-1]=0; A[end,end] = 1;    f[end] = 0

f[end] = 0; f[end] = 0;
ff = [0*f; f]

eye = spdiagm(ones(N+1))
zero_block = zeros(size(A))

S = [-A -eye*h; zero_block A];
uh = S \ ff

ex(x) = (x.^4-2*x.^3+x)/24

th = Vector(0:h:1)
dh = uh[1:N+1]
mh = uh[N+2:end]




