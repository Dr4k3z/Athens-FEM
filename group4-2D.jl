using gmsh
using GR 
using LinearAlgebra
using SparseArrays 
using Plots
using LaTeXStrings

using LinearAlgebra
using SparseArrays 
using StructArrays
using StaticArrays
using StaticRanges
using FastGaussQuadrature

using IterativeSolvers
using Preconditioners

using BenchmarkTools
using Profile
using ProfileView

using Plots

function generateMeshFile(resolution)
    #..1/4: initialize gmsh 
    gmsh.initialize()

    #..2/4: generate geometry 
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("square")
    #..set mesh density parameter 
    lc = 0.02
    #..define four points via (x,y,z) coordinates 
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1.0, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1.0, 1.0, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 1.0, 0, lc, 4)
    #..define four edges by connecting point labels pairwise  
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    #..define curved loop by connecting four edge labels  
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    #..define surface by curved loop 
    gmsh.model.geo.addPlaneSurface([1], 1)
    #..give label to the surface 
    #..syntax of the function being used is gmsh.model.setPhysicalName(dim, tag, name) 
    gmsh.model.setPhysicalName(2, 1, "My surface")
    #..synchronize model 
    gmsh.model.geo.synchronize()


    #..3/4: generate two-dimensional mesh 
    gmsh.model.mesh.generate(2)
    #..if true, write mesh to file for further processing 
    if (true)
        gmsh.write("square.msh")
    end
    #..if true, visualize mesh through the GUI 
    if (false)
        gmsh.fltk.run()
    end

    #..4/4: finalize gmsh 
    gmsh.finalize()
end

#generateMeshFile(1)

function area_triangle(x1,x2,x3,y1,y2,y3)
    x12 = x2 - x1; x13 = x3-x1;
    y12 = y2 - y1; y13 = y3-y1;
    area_id = x12*y13 - x13*y12; 
    area_id = abs(area_id)/2
    return area_id 
end


function meshInfoSetup()
    #..1/12: Finalize gmsh
    gmsh.initialize()

    #..2/12: Generate the mesh
    gmsh.open("square.msh")

    #..3/12 Get and sort the mesh nodes
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

    #..4/12 Get the mesh elements
    #..observe that we get all the two-dimensional triangular elements from the mesh
    element_types, element_ids, element_connectivity = gmsh.model.mesh.getElements(2)
    nelements = length(element_ids[1])
    nelements

    gmsh.finalize()
    return xnode, ynode, node_ids, node_coord, nelements, element_types, element_ids, element_connectivity, nnodes
end
#meshInfoSetup()
function sourceFunctionSetup(xnode, ynode)
    #..5/12 Define the source function and apply the source function to the local coordinates
    #sourcefunction(x, y) = 4 * (x - x^2) + 4 * (y - y^2)
    sourcefunction(x, y) = x + y
    fvalues = map(sourcefunction, xnode, ynode)
    #display(xnode)
    #display(fvalues)
    return fvalues
end

function cvaluesSetup(xnode, ynode)
    densityfunction(x, y) = y
    cvalues = map(densityfunction, xnode, ynode)
    #display(xnode)
    #display(cvalues)
    return cvalues
end

function genLocStiffMat(node_vec,cvalues, xnode, ynode)
    #....retrieve the x and y coordinates of the local nodes of the current element
    xnode1 = xnode[node_vec[1]]; xnode2 = xnode[node_vec[2]]; xnode3 = xnode[node_vec[3]];
    ynode1 = ynode[node_vec[1]]; ynode2 = ynode[node_vec[2]]; ynode3 = ynode[node_vec[3]];
    area_id = area_triangle(xnode1,xnode2,xnode3,ynode1,ynode2,ynode3)
    cavg = mean(cvalues[node_vec])
    Xmat = SMatrix{3,3}(xnode1, xnode2, xnode3, ynode1, ynode2, ynode3, 1, 1, 1) 
    rhs  = SMatrix{3,3}(1., 0., 0., 0., 1., 0., 0., 0., 1.) 
    Emat = MMatrix{3,3}(Xmat\rhs);
    Emat[3,:] .= 0.;  
    Amat = SMatrix{3,3}(cavg*area_id*(transpose(Emat)*Emat));
    Aloc = [Amat[1,:] ; Amat[2,:] ; Amat[3,:] ] 
    Iloc = SVector(node_vec[1],node_vec[2],node_vec[3],node_vec[1],node_vec[2],node_vec[3],node_vec[1],node_vec[2],node_vec[3])
    Jloc = SVector(node_vec[1],node_vec[1],node_vec[1],node_vec[2],node_vec[2],node_vec[2],node_vec[3],node_vec[3],node_vec[3])
    return Iloc, Jloc, Aloc
end

function genStiffMat(cvalues, nelements, element_connectivity, xnode, ynode)

    dofperelem = 9
    #..preallocate the memory for local matrix contributions 
    Avalues = zeros(Float64,dofperelem*nelements)
    I = zeros(Int64,length(Avalues))
    J = zeros(Int64,length(Avalues))
    #display(length(I))


    for element_id in 1:nelements
        node1_id = element_connectivity[1][3*(element_id-1)+1]
        node2_id = element_connectivity[1][3*(element_id-1)+2]
        node3_id = element_connectivity[1][3*(element_id-1)+3]
        nodes = Vector{Int64}([node1_id; node2_id; node3_id])

        
        Iloc, Jloc, Aloc = genLocStiffMat(nodes,cvalues, xnode, ynode) 
        irng = mrange(dofperelem*element_id-dofperelem + 1,dofperelem*element_id) 

        I[irng] .= Iloc 
        J[irng] .= Jloc 
        Avalues[irng] .= Aloc  
    end

    A = sparse(I,J,Avalues)
    return A;
    
end
#S = genStiffMat(cvalues)

function genLocMassMat(node_vec, xnode, ynode)
    #....retrieve the x and y coordinates of the local nodes of the current element
    xnode1 = xnode[node_vec[1]]; xnode2 = xnode[node_vec[2]]; xnode3 = xnode[node_vec[3]];
    ynode1 = ynode[node_vec[1]]; ynode2 = ynode[node_vec[2]]; ynode3 = ynode[node_vec[3]];
    area_id = area_triangle(xnode1,xnode2,xnode3,ynode1,ynode2,ynode3)
    val = area_id/3
    #display(val)
    Aloc = SVector(val,val,val)
    #iter = [node_vec,node_vec,node_vec]
    Iloc = SVector(node_vec[1],node_vec[2],node_vec[3])
    Jloc = SVector(node_vec[1],node_vec[2],node_vec[3])
    #Iloc = node_vec
    #Jloc = node_vec
    #display(Aloc)
    #display(Iloc)
    #display(Jloc)
    return Iloc, Jloc, Aloc
end

function genMassMat(nelements, element_connectivity, xnode, ynode)

    dofperelem = 3
    #..preallocate the memory for local matrix contributions 
    Avalues = zeros(Float64,dofperelem*nelements)
    I = zeros(Int64,length(Avalues))
    J = zeros(Int64,length(Avalues))
    #display(length(I))


    for element_id in 1:nelements
        node1_id = element_connectivity[1][3*(element_id-1)+1]
        node2_id = element_connectivity[1][3*(element_id-1)+2]
        node3_id = element_connectivity[1][3*(element_id-1)+3]
        nodes = Vector{Int64}([node1_id; node2_id; node3_id])

        
        Iloc, Jloc, Aloc = genLocMassMat(nodes, xnode, ynode) 
        irng = mrange(dofperelem*element_id-dofperelem + 1,dofperelem*element_id) 

        I[irng] .= Iloc 
        J[irng] .= Jloc 
        Avalues[irng] .= Aloc  
    end

    A = sparse(I,J,Avalues)
    return A;
    
end

function genLocVec(node_vec, fvalues, xnode, ynode)
    xnode1 = xnode[node_vec[1]]; xnode2 = xnode[node_vec[2]]; xnode3 = xnode[node_vec[3]];
    ynode1 = ynode[node_vec[1]]; ynode2 = ynode[node_vec[2]]; ynode3 = ynode[node_vec[3]];
    area_id = area_triangle(xnode1,xnode2,xnode3,ynode1,ynode2,ynode3)
    Iloc = SVector(node_vec[1],node_vec[2],node_vec[3])
    fval = area_id/3*fvalues[node_vec]
    floc = SVector(fval[1],fval[2],fval[3])
    return Iloc, floc
end

function genVector(fvalues, nelements, nnodes, element_connectivity, xnode, ynode)
    
    
    
    #..initialize global vector  
    f = zeros(Float64,nnodes)

    for element_id = 1:nelements #..loop over number of elements..
        node1_id = element_connectivity[1][3*(element_id-1)+1]
        node2_id = element_connectivity[1][3*(element_id-1)+2]
        node3_id = element_connectivity[1][3*(element_id-1)+3]
        nodes = Vector{Int64}([node1_id; node2_id; node3_id])
        Iloc, floc = genLocVec(nodes, fvalues, xnode, ynode)
        f[Iloc] .+= floc 
    end
   
    return f; 
end

#f = genVector(mesh,fsource)
#println(f)
#f = genVector(fvalues)

#..8/12 Handle the boundary conditions
#..retrieve boundary nodes by loop over corner point and boundary edges
function boundarySetup(A, f)
    gmsh.initialize()
    gmsh.open("square.msh")
    node_ids1=[]; node_ids2=[]; node_ids3=[]; node_ids4=[]; 
    node_ids5=[]; node_ids6=[]; node_ids7=[]; node_ids8=[]; 
    node_ids1, node_coord, _ = gmsh.model.mesh.getNodes(0,1)
    node_ids2, node_coord, _ = gmsh.model.mesh.getNodes(0,2)
    node_ids3, node_coord, _ = gmsh.model.mesh.getNodes(0,3)
    node_ids4, node_coord, _ = gmsh.model.mesh.getNodes(0,4)
    node_ids5, node_coord, _ = gmsh.model.mesh.getNodes(1,1)
    node_ids6, node_coord, _ = gmsh.model.mesh.getNodes(1,2)
    node_ids7, node_coord, _ = gmsh.model.mesh.getNodes(1,3)
    node_ids8, node_coord, _ = gmsh.model.mesh.getNodes(1,4)
    bnd_node_ids = union(node_ids1,node_ids2,node_ids3,node_ids4,node_ids5,node_ids6,node_ids7,node_ids8)
    A[bnd_node_ids,:] .= 0;
    A[bnd_node_ids,bnd_node_ids] = Diagonal(ones(size(bnd_node_ids)))
    f[bnd_node_ids] .= 0;

    gmsh.finalize()
end

#boundarySetup(A, f)
#..9/12 Make A sparse and M diagonal 
#A = sparse(A)
#M = Diagonal(M)

#..10/12 Compute the numerical solution
#u = A\f

#..10/12: Finalize gmsh


#..11/12 Plot the numerical solution
#GR.trisurf(xnode,ynode,u)


function buildMatandVec_2D(resolution)
    
    generateMeshFile(resolution)
    xnode, ynode, node_ids, node_coord, nelements, element_types, element_ids, element_connectivity, nnodes = meshInfoSetup()
    
    fvalues = sourceFunctionSetup(xnode, ynode)
    cvalues = cvaluesSetup(xnode, ynode)
    
    
    elapsed_time_assemblying = @elapsed begin
        S = genStiffMat(cvalues, nelements, element_connectivity, xnode, ynode)
        #@time genStiffMat(cvalues, nelements, element_connectivity, xnode, ynode)
        M = genMassMat(nelements, element_connectivity, xnode, ynode)
        #@time genMassMat(nelements, element_connectivity, xnode, ynode)
        f = genVector(fvalues, nelements, nnodes, element_connectivity, xnode, ynode)
        #@time genVector(fvalues, nelements, nnodes, element_connectivity, xnode, ynode)
        A = (S - M)
        boundarySetup(A, f)
    end
    returning = [A,f,elapsed_time_assemblying]

    return returning;
end



