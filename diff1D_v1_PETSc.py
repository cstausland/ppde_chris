# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc
# should implement setFromOptions for solver type

def solver_FE_PETSc(I, a, L, Nx, C, T):
	
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
	u, u_1 = A.getVecs()
	
	# Initialize first time step from the numpy array I
	u_1.setValues(range(Nx+1),I)
	
	# Assemble the vectors and matrix. This partition each part
	# on its correct process/rank
	A.assemblyBegin(); A.assemblyEnd()
	u.assemblyBegin(); u.assemblyEnd()
	u_1.assemblyBegin(); u_1.assemblyEnd()
	
	# This can be cut out, this is for plotting in external
	# programs
	if PETSc.Options().getBool('toFile',default=False):
		if PETSc.COMM_WORLD.getSize() > 1:
			if PETSc.COMM_WORLD.getRank() == 0:
				print 'Warning: not writing to file (using parallel)'
		else:
			print 'Writing to file'
			W = PETSc.Viewer().createASCII('test3.txt',format=1)
			u_1.view(W)
	
	for n in range(0, Nt):
		# Solve for next step. This should be done without
		# matrix multiplication (solving the system directly).
		A.mult(u_1, u)
		
		# Copy new solution to the old vector
		u.copy(u_1)
		
		# Set boundary condition
		u_1.setValue(0,0); u_1.setValue(Nx,0)
		
		if PETSc.Options().getBool('toFile',default=False):
			u_1.view(W)
	return time.clock()-t0

def solver_BE_PETSc(I, a, L, Nx, C, T):
	
	t0 = time.clock()
	x = np.linspace(0, L, Nx+1)
	dx = x[1] - x[0]
	dt = C*dx**2/a
	Nt = int(round(T/float(dt)))
	t = np.linspace(0, T, Nt+1)
	
	A = PETSc.Mat().createAIJ([Nx+1, Nx+1],nnz=3)
	for i in range(1, Nx+1):   # filling the diagonal and off-diagonal entries
		A.setValue(i,i,1.+2*C)
		A.setValue(i-1,i,-C)
		A.setValue(i,i-1,-C)
	# For boundary conditions
	A.setValue(0,0,1); A.setValue(0,1,0);
	A.setValue(Nx,Nx,1); A.setValue(Nx,Nx-1,0);
	
	# Create the to vectors, we need two of them (new and old). We're also
	# making a temporary RHS-array b
	u, u_1 = A.getVecs()
	
	# Assemble the matrix and vectors. This distribute the data on to the
	# different processors (among other things).
	A.assemblyBegin(); A.assemblyEnd()
	u.assemblyBegin(); u.assemblyEnd()
	u_1.assemblyBegin(); u_1.assemblyEnd()
	
	if PETSc.Options().getBool('toFile',default=False):
		if PETSc.COMM_WORLD.getSize() > 1:
			if PETSc.COMM_WORLD.getRank() == 0:
				print 'Warning: not writing to file (using parallel)'
		else:
			print 'Writing to file'
			W = PETSc.Viewer().createASCII('test3.txt',format=1)
			u_1.view(W)
	
	"""
	Setting up the KSP solver, the heart of PETSc. We use setFromOptions,
	and some of the	command line options is available at:
	http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetFromOptions.html
	
	The command line option for the KSP solver is -ksp_type <methodname>,
	and PETSc manual Table 3 list the method names.
	
	The command line option for preconditioner is -pc_type <methodname>,
	and the available method names is listed in Table 4 in the PETSc manual.
	
	As an example: to use a direct solver method using LU, run the program using:
		-ksp_type preonly -pc_type lu
	To run with a Conjugate Gradient iterativ solver (nice for positive-definite and
	symmetric matrices often seen in Finite Difference systems) with incomplete
	Cholesky preconditioning, run with:
		-ksp_type cg -pc_type icc
	
	The solvers and preconditioners are also listed at (along with external packages):
	http://www.mcs.anl.gov/petsc/petsc-current/docs/linearsolvertable.html
	"""
	
	ksp = PETSc.KSP()
	ksp.create()
									# Setting a CG iterative solver as default, then one can override
									# this with command line options:
	ksp.setType('cg')				# KSP solver
	ksp.getPC().setType('icc')		# Preconditioner (LU factorization)
	
	ksp.setFromOptions()
	ksp.setOperators(A)				# Set which matrix to solve the problem with
	
	opt = PETSc.Options()
	if PETSc.COMM_WORLD.getRank() == 0:
		print 'For BE:\n   KSP type: ',ksp.getType(),'  PC type: ', ksp.getPC().getType(),'\n'
	
	for t in range(Nt):
		# Solve for next step using the solver set up above using
		# command-line options.
		ksp.solve(u_1,u)
		
		# Could we use maybe use this:   u, u_1 = u_1, u
		# to prevent hard-copying?
		u.copy(u_1)
		
		if PETSc.Options().getBool('toFile',default=False):
			u.view(W)
	
	return time.clock()-t0
#


# Run using e.g.:
# /usr/lib64/openmpi/1.4-gcc/bin/mpirun -np 1 python2.6 diff1D_v1_PETSc.py -ksp_type cg -pc_type icc -Nx 100 -T 0.5

a = PETSc.Options().getReal('a',default=1.0)
L = PETSc.Options().getReal('L',default=1.0)
Nx = PETSc.Options().getInt('Nx',default=100)
C = PETSc.Options().getReal('C',default=0.5)
T = PETSc.Options().getReal('T',default=0.25)

x = np.linspace(0,L,Nx+1)
I = np.exp(-np.square(x-0.5*L)*50)

t_time_fe = solver_FE_PETSc(I, a, L, Nx, C, T)
t_time_be = solver_BE_PETSc(I, a, L, Nx, C, T)

if PETSc.COMM_WORLD.getRank() == 0:
	print 'Timings:\n   FE: ', t_time_fe, '    BE: ', t_time_be

#if PETSc.COMM_WORLD.getRank() == 0:
#	print 'Timings:\n   FE: ', t_time_fe

# Seems like CG uses twice the time as LU