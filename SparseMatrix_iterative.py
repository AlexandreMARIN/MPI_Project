#!/bin/env python
# -*- coding: utf-8 -*-
import mesh
import fem
import laplacian
import splitter
from math import cos,sin,pi,sqrt
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import VisuSplitMesh as VSM
import VisuSolution as VS
from scipy.sparse import linalg as sp_linalg
from mpi4py import MPI
from conjugate_gradient import *


def g(x,y) :
    return cos(2*pi*x)+sin(2*pi*y)




# Duplication du communicateur :
comm = MPI.COMM_WORLD.Dup()
# Interrogation du contexte MPI :
rank = comm.rank
nbDoms = comm.size #nombre de domaines

# Ouverture d'un fichier nom unique mode simple
fileName = "sortie{}.txt".format(rank)

file = open(fileName, mode="w")
file.write("Rang du processus : {}\n".format(rank))
file.write("Nombre de processus : {}\n".format(nbDoms))


#Process 0 reads the mesh, builds data structures and send them to processes 1,...,nbDoms-1
if rank == 0:
    m = mesh.read("CarreGros.msh")
    coords    = m[0]
    elt2verts = m[1]
    nbVerts = coords.shape[0]
    nbElts  = elt2verts.shape[0]
    print('nbVerts : {}'.format(nbVerts))
    print('nbElts  : {}'.format(nbElts))
    etsDomains = splitter.element( nbDoms, (elt2verts, coords) )
    print("Nombre d'elements par domaines : ")
    i = 0
    for a in etsDomains :
        print("Domaine {} : {} elements\n".format(i,a.shape[0]))
        i += 1

    elt2doms = np.zeros((nbElts,), np.double)
    ia = 0
    for a in etsDomains :
        for e in a :
            elt2doms[e] = ia
        ia += 1

    # Calcule l'interface :
    ie = 0
    mask = np.array([-1,]*nbVerts, np.short)
    for e in elt2verts :
        d = elt2doms[ie]
        if mask[e[0]] == -1 :
            mask[e[0]] = d
        elif mask[e[0]] != d :
            mask[e[0]] = -2
        if mask[e[1]] == -1 :
            mask[e[1]] = d
        elif mask[e[1]] != d :
            mask[e[1]] = -2
        if mask[e[2]] == -1 :
            mask[e[2]] = d
        elif mask[e[2]] != d :
            mask[e[2]] = -2
        ie += 1
        
    nbInterf = 0
    for m in mask :
        if m == -2 :
            nbInterf += 1

    interfNodes = np.empty(nbInterf, np.long)
    nbInterf = 0
    for im in range(mask.shape[0]):
        if mask[im] == -2 :
            interfNodes[nbInterf] = im
            nbInterf += 1

    VSM.view( coords, elt2verts, nbDoms, elt2doms, indInterfNodes = interfNodes, title='Partition par elements')
    comm.bcast((nbVerts, nbElts), root = 0)
    comm.Bcast(coords, root = 0)
    comm.Bcast(elt2verts, root = 0)
    ets_loc2glo = comm.scatter(etsDomains, root = 0)
    #ets_loc2glo associates indices of local elements with indices of elements in the global mesh
else:
    #we receive information
    nbVerts, nbElts = comm.bcast(None, root = 0)
    coords = np.empty((nbVerts, 4))
    comm.Bcast(coords, root = 0)
    elt2verts = np.empty((nbElts, 3), dtype=np.int64)
    comm.Bcast(elt2verts, root = 0)
    ets_loc2glo = comm.scatter(None, root = 0)


#we build here the local mesh:

#e2v_loc : equivalent to elt2verts but for the local mesh
e2v_loc = np.empty((ets_loc2glo.shape[0], 3), dtype=np.int64)
v_loc = set()
for ie in ets_loc2glo:
    v_loc.add(elt2verts[ie, 0])
    v_loc.add(elt2verts[ie, 1])
    v_loc.add(elt2verts[ie, 2])

v_loc2glo = sorted(v_loc)#associates local indices of vertices to global indices of vertices
v_glo2loc = np.array([-1]*coords.shape[0])#this is the inverse map of v_loc2glo

for ind in range(len(v_loc2glo)):
    v_glo2loc[v_loc2glo[ind]] = ind

#we give values for e2v_loc
for ie_loc in range(len(ets_loc2glo)):
    ie = ets_loc2glo[ie_loc]
    e2v_loc[ie_loc, 0] = v_glo2loc[elt2verts[ie, 0]]
    e2v_loc[ie_loc, 1] = v_glo2loc[elt2verts[ie, 1]]
    e2v_loc[ie_loc, 2] = v_glo2loc[elt2verts[ie, 2]]


#structure for the sparse local matrix
begVert2Elts_loc, vert2elts_loc = mesh.compvert2elts(e2v_loc)

begRows_loc, indCols_loc = fem.comp_skel_csr_mat(e2v_loc, (begVert2Elts_loc, vert2elts_loc) )
nz = begRows_loc[-1]
file.write("\nNumber of non zero in sparse matrix : {}".format(nz))

#number of vertices and number of elements in the local mesh
nbV_loc = len(v_loc2glo)
nbE_loc = e2v_loc.shape[0]

spCoefs = np.zeros( (nz,), np.double)#coefficients of the sparse matrix

#we add elementary matrices, only for each element in the local mesh
for iElt in range(nbE_loc):
    iVertices = e2v_loc[iElt,:]
    crd1 = coords[v_loc2glo[iVertices[0]],:]
    crd2 = coords[v_loc2glo[iVertices[1]],:]
    crd3 = coords[v_loc2glo[iVertices[2]],:]
    matElem = laplacian.comp_eltmat((crd1[0],crd1[1]), (crd2[0],crd2[1]),
                                    (crd3[0],crd3[1]))
    fem.add_elt_mat_to_csr_mat((begRows_loc,indCols_loc,spCoefs),
                               (iVertices, iVertices, matElem))


# Assemblage second membre :
f = np.zeros(nbV_loc, np.double)
for iVert in range(nbV_loc):
    iVGlo = v_loc2glo[iVert]
    if ( coords[iVGlo,3] > 0 ) :
        f[iVert] += g(coords[iVGlo,0],coords[iVGlo,1])
b = np.zeros(nbV_loc, np.double)
for i in range(nbV_loc) :
    for ptR in range(begRows_loc[i],begRows_loc[i+1]):
        b[i] -= spCoefs[ptR]*f[indCols_loc[ptR]]


#we compute 'bdry', intersection of the boundary of the global mesh and of the boundary of the local mesh
bdry = set() #it will contain global indices

for iV in range(nbV_loc):
    iVglo = v_loc2glo[iV]
    if coords[iVglo, 3] > 0:
        bdry.add(iVglo)

#we compute the interface (with other local meshes)
edges = set()#boundary of the local mesh, global indices are considered
for ie in range(nbE_loc):
    ieglo = ets_loc2glo[ie]
    for i in range(3):
        edge = None
        v1 = elt2verts[ieglo, i%3]
        v2 = elt2verts[ieglo, (i+1)%3]
        if v1 < v2:
            edge = (v1, v2)
        else:
            edge = (v2, v1)
        if edge in edges:
            edges.remove(edge)
        else:
            edges.add(edge)

interfNodes = set() #this variable will contain nodes at the interface of the local mesh (except those in the boundary of the global mesh)
for e in edges:
    interfNodes.add(e[0])
    interfNodes.add(e[1])

interfNodes = interfNodes - bdry #we process differently nodes of the interface which are also in the boundary of the global mesh

#we ensure each node in the interface will be considered by only one process (for the scalar product function, see below)
interfNodesToDiscard = set()
file.write("\n\ninterfNodes:\n"+str(interfNodes)+"\n")
for ip in range(rank):
    interfNodes2 = comm.recv(source=ip)
    interfNodesToDiscard = interfNodesToDiscard | (interfNodes2 & interfNodes)
file.write("interfNodesToDiscard:\n"+str(interfNodesToDiscard)+"\n")#"not to consider"
for ip in range(rank+1, nbDoms):
    comm.send(interfNodes, dest=ip)


#now for the boundary conditions, nodes in the boundary AND in the interfaces must be associated with only one process
#we can do better by defining more comms
toNullify = set() #it will contain global indices of nodes in the local interface AND in the global boundary
#thus, the matching line in the global matrix will be correct

#too complex
#subcomm = []
#for ip in range(rank+1):
#    subcomm.append(comm.Split(color=ip, key=rank))
#for ip in range(rank+1, nbDoms):
#    comm.Split(color=MPI.UNDEFINED, key=rank)

#that method works:
for ip in range(rank):
    bdry2 = comm.recv(source=ip)
    toNullify = toNullify | (bdry & bdry2)
file.write("toNullify:\n"+str(toNullify)+"\n")
for ip in range(rank+1, nbDoms):
    comm.send(bdry, dest=ip)


# Il faut maintenant tenir compte des conditions limites :
for iVert in bdry:
    iVloc = v_glo2loc[iVert]
    # Suppression de la ligne avec 1 sur la diagonale :
    for i in range(begRows_loc[iVloc],begRows_loc[iVloc+1]):
        if indCols_loc[i] != iVloc or iVert in toNullify:
            spCoefs[i] = 0.
        else :
            spCoefs[i] = 1.
    # Suppression des coefficients se trouvant sur la colonne iVert :
    for iRow in range(nbV_loc):
        if iRow != iVloc :
            for ptCol in range(begRows_loc[iRow],begRows_loc[iRow+1]):
                if indCols_loc[ptCol] == iVloc :
                    spCoefs[ptCol] = 0.
                        
    b[iVloc] = f[iVloc]

# On definit ensuite la matrice :
spMatrix = sparse.csc_matrix((spCoefs, indCols_loc, begRows_loc),
                             shape=(nbV_loc,nbV_loc))

#file.write("\nMatrice creuse {}:\n{}\n\n".format(spMatrix.shape, spMatrix))
#file.write("\nb_loc:\n"+str(b)+"\n")

#we build a part of global b (no problem due to overlapping)
partOfbGlo = np.zeros(nbVerts)
for iVloc in set(range(nbV_loc)):
    iVglo = v_loc2glo[iVloc]
    if iVglo not in toNullify:
        partOfbGlo[iVglo] = b[iVloc]

partsOfb = comm.allgather(partOfbGlo)
bGlo = np.zeros(nbVerts) #will contain the global second member
for pob in partsOfb:
    bGlo += pob


# Visualisation second membre :
#VS.view( coords, elt2verts, b, title = "second membre" )


def prodMV(x):
    """
    That function computes the product of the global matrix by the vector 'x'
    Each process handles its local sparse matrix.
    """
    x_loc = np.zeros(nbV_loc)
    for iVloc in range(nbV_loc):
        iVglo = v_loc2glo[iVloc]
        x_loc[iVloc] = x[iVglo]
    y_loc = spMatrix.dot(x_loc)
    partOfy = np.zeros(nbVerts)
    for iVloc in range(nbV_loc):
        iVglo = v_loc2glo[iVloc]
        partOfy[iVglo] = y_loc[iVloc]

    recvbuf = np.empty(nbVerts*nbDoms)
    comm.Allgather(sendbuf=partOfy, recvbuf=recvbuf)
    for ip in range(rank):
        partOfy += recvbuf[ip*nbVerts:(ip+1)*nbVerts]
    for ip in range(rank+1, nbDoms):
        partOfy += recvbuf[ip*nbVerts:(ip+1)*nbVerts]
    return partOfy


#we compute global indices of nodes to consider for the scalar product
nodesToConsider = set()
for iVloc in range(nbV_loc):
    iVglo = v_loc2glo[iVloc]
    nodesToConsider.add(iVglo)

nodesToConsider -= (interfNodesToDiscard | toNullify)

def scalarProd(x, y):
    """
    That function computes the scalar product of 'x' by 'y'.
    Each coordinate of 'x' and 'y' is associated with only one process.
    """
    partOfRes = 0
    for iV in nodesToConsider:
        partOfRes += x[iV]*y[iV]
    partsOfRes = comm.allgather(partOfRes)
    for ip in range(rank):
        partOfRes += partsOfRes[ip]
    for ip in range(rank+1, nbDoms):
        partOfRes += partsOfRes[ip]
    return partOfRes


#resolution
sol, info = solver_gc(None, bGlo, prodMatVect=prodMV, verbose=(rank==0), prodScal=scalarProd)

if rank==0:
    #print("sol: ", sol)
    # Visualisation de la solution :
    VS.view( coords, elt2verts, sol, title = "Solution" )

    
