import gmsh
using GR

using LinearAlgebra
using SparseArrays
using StaticArrays
using StaticRanges

using BenchmarkTools

using Test

using Plots

# struct to hold 2D point
struct Point
    x::Float64   # x coordinates
    y::Float64   # y coordinates 
end

# struct to hold a single mesh element
struct Element
    p1::Point       # coordinates first node 
    p2::Point       # coordinates second node 
    p3::Point       # coordinates third node     
    e1::Int64       # global index first node
    e2::Int64       # global index second node
    e3::Int64       # global index third node
    area::Float64   # area of the element 
end

# struct to hold entire mesh
struct Mesh
    nnodes::Int64               # number of nodes 
    nelems::Int64               # number of elements
    Elements::Array{Element,1}  # list of Elements 
    bndNodeIds::Vector{Int64}   # indices of nodes where Dirichlet bc are applied  
    dofPerElem::Int64           # number of dofs per element 
end

# function area_triangle(p1::Point,p2::Point,p3::Point)
function area_triangle(p1, p2, p3)
    x12 = p2.x - p1.x
    x13 = p3.x - p1.x
    y12 = p2.y - p1.y
    y13 = p3.y - p1.y
    area_id = x12 * y13 - x13 * y12
    area_id = abs(area_id) / 2.0
    return area_id
end

# read elements from mesh file 
function meshFromGmsh(meshFile)

    #..Initialize GMSH
    gmsh.initialize()

    #..Read mesh from file
    gmsh.open(meshFile)

    #..Get the mesh nodes
    #..Observe that although the mesh is two-dimensional,
    #..the z-coordinate that is equal to zero is stored as well.
    #..Observe that the coordinates are stored contiguously for computational efficiency
    node_ids, node_coord, _ = gmsh.model.mesh.getNodes()
    nnodes = length(node_ids)
    #..sort the node coordinates by ID, such that Node one sits at row 1
    tosort = [node_ids node_coord[1:3:end] node_coord[2:3:end]]
    sorted = sortslices(tosort, dims=1)
    node_ids = sorted[:, 1]
    xnode = sorted[:, 2]
    ynode = sorted[:, 3]

    #..Get the mesh elements
    #..Observe that we get all the two-dimensional triangular elements from the mesh
    element_types, element_ids, element_connectivity = gmsh.model.mesh.getElements(2)
    nelems = length(element_ids[1])

    #..Construct uninitialized array of length nelements  
    Elements = Array{Element}(undef, nelems)

    #..Construct the array of elements 
    for element_id in 1:nelems
        e1 = element_connectivity[1][3*(element_id-1)+1]
        e2 = element_connectivity[1][3*(element_id-1)+2]
        e3 = element_connectivity[1][3*(element_id-1)+3]
        p1 = Point(sorted[e1, 2], sorted[e1, 3])
        p2 = Point(sorted[e2, 2], sorted[e2, 3])
        p3 = Point(sorted[e3, 2], sorted[e3, 3])
        area = area_triangle(p1, p2, p3)
        Elements[element_id] = Element(p1, p2, p3, e1, e2, e3, area)
    end

    #..retrieve boundary nodes by loop over corner point and boundary edges
    node_ids1 = []
    node_ids2 = []
    node_ids3 = []
    node_ids4 = []
    node_ids5 = []
    node_ids6 = []
    node_ids7 = []
    node_ids8 = []
    node_ids1, node_coord, _ = gmsh.model.mesh.getNodes(0, 1)
    node_ids2, node_coord, _ = gmsh.model.mesh.getNodes(0, 2)
    node_ids3, node_coord, _ = gmsh.model.mesh.getNodes(0, 3)
    node_ids4, node_coord, _ = gmsh.model.mesh.getNodes(0, 4)
    node_ids5, node_coord, _ = gmsh.model.mesh.getNodes(1, 1)
    node_ids6, node_coord, _ = gmsh.model.mesh.getNodes(1, 2)
    node_ids7, node_coord, _ = gmsh.model.mesh.getNodes(1, 3)
    node_ids8, node_coord, _ = gmsh.model.mesh.getNodes(1, 4)
    bnd_node_ids = union(node_ids1, node_ids2, node_ids3, node_ids4, node_ids5, node_ids6, node_ids7, node_ids8)

    #..Set DOF per element
    dofPerElement = 9

    #..Store data inside mesh struct  
    mesh = Mesh(nnodes, nelems, Elements, bnd_node_ids, dofPerElement)

    #..Finalize gmsh
    gmsh.finalize()

    return mesh
end

#..read nodes from mesh file (useful for post-processing)
function nodesFromGmsh(meshFile)

    #..Initialize GMSH
    gmsh.initialize()

    #..Read mesh from file
    gmsh.open(meshFile)

    #..Get the mesh nodes
    #..Observe that although the mesh is two-dimensional,
    #..the z-coordinate that is equal to zero is stored as well.
    #..Observe that the coordinates are stored contiguously for computational
    #..efficiency
    node_ids, node_coord, _ = gmsh.model.mesh.getNodes()
    nnodes = length(node_ids)
    #..sort the node coordinates by ID, such that Node one sits at row 1
    tosort = [node_ids node_coord[1:3:end] node_coord[2:3:end]]
    sorted = sortslices(tosort, dims=1)
    node_ids = sorted[:, 1]
    xnode = sorted[:, 2]
    ynode = sorted[:, 3]

    #..Finalize gmsh
    gmsh.finalize()

    return xnode, ynode
end

function genLocStiffMat(element::Element)
    p1 = element.p1
    p2 = element.p2
    p3 = element.p3
    e1 = element.e1
    e2 = element.e2
    e3 = element.e3
    area = element.area
    Iloc = SVector(e1, e1, e1, e2, e2, e2, e3, e3, e3)
    Jloc = SVector(e1, e2, e3, e1, e2, e3, e1, e2, e3)
    Xmat = SMatrix{3,3}(p1.x, p2.x, p3.x, p1.y, p2.y, p3.y, 1, 1, 1)
    rhs = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    Emat = MMatrix{3,3}(Xmat \ rhs)
    Emat[3, :] .= 0.0
    Amat = SMatrix{3,3}(area * (transpose(Emat) * Emat))
    Aloc = [Amat[1, :]; Amat[2, :]; Amat[3, :]]
    return Iloc, Jloc, Aloc
end

function genStiffMat(mesh::Mesh)

    #..recover number of elements  
    nelems = mesh.nelems
    dofPerElem = mesh.dofPerElem

    #..preallocate the memory for local matrix contributions 
    Avalues = zeros(Float64, dofPerElem * nelems)
    I = zeros(Int64, length(Avalues))
    J = zeros(Int64, length(Avalues))

    for i = 1:nelems #..loop over number of elements..
        element = mesh.Elements[i]
        Iloc, Jloc, Aloc = genLocStiffMat(element)
        irng = mrange(dofPerElem * i - 8, dofPerElem * i)
        I[irng] = Iloc
        J[irng] = Jloc
        Avalues[irng] = Aloc
    end

    A = sparse(I, J, Avalues)

    return A
end

function genLocMassMat(element::Element)
    p1 = element.p1
    p2 = element.p2
    p3 = element.p3
    e1 = element.e1
    e2 = element.e2
    e3 = element.e3
    area = element.area
    Iloc = SVector(e1, e1, e1, e2, e2, e2, e3, e3, e3)
    Jloc = SVector(e1, e2, e3, e1, e2, e3, e1, e2, e3)
    Mloc = SMatrix{3,3}(area / 3, 0.0, 0.0, 0.0, area / 3, 0.0, 0.0, 0.0, area / 3)
    return Iloc, Jloc, Mloc
end

function genMassMat(mesh::Mesh)

    #..recover number of elements  
    nelems = mesh.nelems
    dofPerElem = mesh.dofPerElem

    #..preallocate the memory for local matrix contributions 
    Mvalues = zeros(Float64, dofPerElem * nelems)
    I = zeros(Int64, length(Mvalues))
    J = zeros(Int64, length(Mvalues))

    for i = 1:nelems #..loop over number of elements..
        element = mesh.Elements[i]
        Iloc, Jloc, Mloc = genLocMassMat(element)
        irng = mrange(dofPerElem * i - 8, dofPerElem * i)
        I[irng] = Iloc
        J[irng] = Jloc
        Mvalues[irng] = Mloc
    end

    M = sparse(I, J, Mvalues)

    return M
end

mySourceFct(x, y) = x + y

function genLocVector(element::Element, sourceFct::Function)
    p1 = element.p1
    p2 = element.p2
    p3 = element.p3
    e1 = element.e1
    e2 = element.e2
    e3 = element.e3
    area = element.area
    Iloc = SVector(e1, e2, e3)
    # use broadcast for the lines below instead 
    f1 = area / 3 * sourceFct(p1.x, p1.y)
    f2 = area / 3 * sourceFct(p2.x, p2.y)
    f3 = area / 3 * sourceFct(p3.x, p3.y)
    floc = SVector(f1, f2, f3)
    return Iloc, floc
end

function genVector(mesh, sourceFct::F) where {F}

    #..recover number of elements  
    nelems = mesh.nelems
    nnodes = mesh.nnodes

    #..preallocate the memory for local matrix contributions 
    f = zeros(Float64, nnodes)

    for i = 1:nelems #..loop over number of elements..
        element::Element = mesh.Elements[i]
        Iloc, floc = genLocVector(element, sourceFct)
        f[Iloc] += floc
    end

    return f
end

function handleBoundary!(mesh, A, f)
    bndNodeIds = mesh.bndNodeIds
    #..handle essential boundary conditions 
    A[bndNodeIds, :] .= 0
    A[bndNodeIds, bndNodeIds] = Diagonal(ones(size(bndNodeIds)))
    f[bndNodeIds] .= 0
    return A, f
end

function generateSolution(mesh, A, f)

    A, f = handleBoundary!(mesh, A, f)
    u = A \ f
    return u
end

mesh = meshFromGmsh("square-100.msh");
A = genStiffMat(mesh);
f = genVector(mesh, mySourceFct);

display(A)

u1 = generateSolution(mesh, A, f)

A, f = handleBoundary!(mesh, A, f);

using Preconditioners
using IterativeSolvers
psA = CholeskyPreconditioner(A, 2)
#@btime CholeskyPreconditioner(sA, 1)
u2 = cg(A, f, Pl=psA);
#@btime cg(sA, f, Pl=psA);

display(u2)