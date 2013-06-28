# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc

def solver_FE_PETSc(I, a, L, Nx, C, T):
	"""
	Simplest expression of the computational algorithm
	using the Forward Euler method and explicit Python loops.
	For this method C <= 0.5 for stability.
	"""
	
	t0 = time.clock()
	x = np.linspace(0, L, Nx+1)
	dx = x[1] - x[0]
	dt = C*dx**2/a
	Nt = int(round(T/float(dt)))
	t = np.linspace(0, T, Nt+1)
	
	A = PETSc.Mat().createAIJ([Nx+1, Nx+1],nnz=3)
	for i in range(1, Nx+1):   # filling the non-zero entires
		A.setValue(i,i,1.-2*C)
		A.setValue(i-1,i,C)
		A.setValue(i,i-1,C)
	A.setValue(0,0,1); A.setValue(0,1,0);
	A.setValue(Nx,Nx,1); A.setValue(Nx,Nx-1,0);
	
	# Get the left and right hand side vectors with properties
	# from the matrix T, e.g. type and sizes.
	u = PETSc.Vec().createMPI(Nx+1); #un.set(1);
	u_1 = PETSc.Vec().createMPI(Nx+1)
	
	# Initialize first time step from the numpy array I
	u_1.setValues(range(Nx+1),I)
	
	# Assemble the vectors and matrix. This partition each part
	# on its correct process/rank
	A.assemblyBegin(); A.assemblyEnd()
	u.assemblyBegin(); u.assemblyEnd()
	u_1.assemblyBegin(); u_1.assemblyEnd()
	
	# This can be cut out, this is for plotting in external
	# programs
	outputToFile = False
	if outputToFile:
		W = PETSc.Viewer().createASCII('test.txt',format=1)
		u_1.view(W)
	
	for n in range(0, Nt):
		# Solve for next step. This should be done without
		# matrix multiplication (solving the system directly).
		A.mult(u_1, u)
		
		# Copy new solution to the old vector
		u.copy(u_1)
		
		# Set boundary condition
		u_1.setValue(0,0); u_1.setValue(Nx,0)
		
		if outputToFile:
			u.view(W)
	return 
#
'''
def solver_BE_PETSc(I, a, L, Nx, C, T):
	t0 = time.clock()
'''


if (len(sys.argv) > 1):
	Nx = int(sys.argv[1])
else:
	Nx = 100

if (len(sys.argv) > 2):
	T = float(sys.argv[2])
else:
	T = 1.

L = 1; a = 1; C = 0.50;
x = np.linspace(0,L,Nx+1)
#I = 3*x**2;
I = np.exp(-np.square(x-0.5*L)*50)

solver_FE_PETSc(I, a, L, Nx, C, T)
#print 'Done'