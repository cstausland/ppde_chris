# Kjoring hos meg:     python2.6 backward_diffusion.py 1000 200

# ------------------------------------------------------------------------
# 
# 1-D diffusion equation using Backward Euler (implicit). The diffusion
# equation is
# 		du/dt = d2u/dx2.
# 
# The initial condition used here is exp(x = -100:100), but this can
# easily be replaced by any function. The boundary conditions is 0 at
# both ends.
# 
# ------------------------------------------------------------------------

# Import and initialize
import petsc4py, sys, numpy
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Spatial points
if (len(sys.argv) > 1):
	n = int(sys.argv[1])
else:
	n = 100

# Temporal points (number of time steps)
if (len(sys.argv) > 2):
	Nt = int(sys.argv[2])
else:
	Nt = 100

print 'Running with', n,'spatial points and',Nt,'time steps'


a_from = 0; b_to = 1; dx = (b_to-a_from)/(n-1.);
C = 100*dx**2/(dx**2)    # C = dt/dx^2

# Need our two matrices. One tridiag and one diag of size n-by-n. These
# matrices are sparse by nature, so we use the default PETSc format AIJ
# which is sparse. One should (not must) preallocate the number of data
# points using nnz. For a tridiagonal matrix nnz=3 and for a diagonal
# matrix nnz=1. nnz is approx. number of elements per row.
T = PETSc.Mat().createAIJ([n, n],nnz=3)
for i in range(1, n):   # filling the diagonal and off-diagonal entries
	T.setValue(i,i,1.+2*C)
	T.setValue(i-1,i,-C)
	T.setValue(i,i-1,-C)
# For boundary conditions
T.setValue(0,0,1); T.setValue(0,1,0);
T.setValue(n-1,n-1,1); T.setValue(n-1,n-2,0);

# The diagonal-mat is a pure eye matrix.
D = PETSc.Mat().createAIJ([n, n],nnz=1)
D.setDiagonal(PETSc.Vec().createWithArray([1]*n))
# For boundary conditions
D.setValue(0,0,0); D.setValue(n-1,n-1,0);

# Create the to vectors, we need two of them (new and old). We're also
# making a temporary RHS-array b
un = PETSc.Vec().createSeq(n)
unm1 = PETSc.Vec().createSeq(n)
b = PETSc.Vec().createSeq(n)

# The inital condition is an exponential or sine for now
#unm1.setValues(range(n),numpy.exp(-numpy.square(numpy.linspace(a_from,b_to,n)-0.5*(b_to-a_from))*50))
#unm1.setValues(range(n),numpy.sin(numpy.linspace(a_from,b_to,n)*2*numpy.pi))
unm1.setValues(range(n),numpy.linspace(a_from,b_to,n))
unm1.setValue(0,0); unm1.setValue(n-1,0);

# Assemble all vectors and matrices. (Is all of this necesarry?)
un.assemblyBegin()
un.assemblyEnd()
unm1.assemblyBegin()
unm1.assemblyEnd()
T.assemblyBegin()
T.assemblyEnd()
D.assemblyBegin()
D.assemblyEnd()


# Set up a solver. Using conjugate gradient (cg) for now (iterative),
# and incomplete Cholesky (icc) as preconditioner
'''
ksp = PETSc.KSP()
ksp.create()
ksp.setType('cg')             # Setting solver type
ksp.getPC().setType('icc')    # Preconditioner
ksp.setOperators(T)           # Set which matrix to solve the problem with
'''

# For a list of preconditioners:
# http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCType.html#PCType
# There seems to be convergence issues for this problem using an
# iterative solver (dependent of n). Using a direct solver solves without
# iteration and one does not get convergence issues. Direct methods tends
# to be slower(?)
#
# Here we uses LU factorization as a direct solver. This is set as a
# preconditioner, and we therefore sets the solver type to be 'preonly'.
ksp = PETSc.KSP()
ksp.create()
ksp.setType('preonly')        # Setting preonly as solver type
ksp.getPC().setType('lu')     # Preconditioner (LU factorization)
ksp.setOperators(T)           # Set which matrix to solve the problem with


plotting = False
# Importing plotting tool
if plotting:
	try:
		from matplotlib import pylab
		pylab.figure()
		x = numpy.linspace(-100,100,n)
	except ImportError:
		print 'WARNING: Matplotlib dont exist on this system, and', \
			'the results will not be plotted'
		plotting = False

outputToFile = True
if outputToFile:
	W = PETSc.Viewer().createASCII('test.txt',format=0)
	unm1.view(W)

# Now solve each time step sequentially
for t in range(Nt):
	# Calculate RHS (b): T*un = b = D*unm1. Since D in this example is the
	# identity matrix, b=unm1
	#D.mult(unm1,b)
	unm1.copy(b)
	
	# Then run the solver. T is presetup and constant. 'un' and 'b'
	# is the only thing that changes for each time step
	ksp.solve(b,un)
	
	# Update unm1 to the new one, then we're ready for another timestep
	un.copy(unm1)
	
	#print t
	#print un.getArray()
	
	# Is this the correct use? (Matlab style)
	if plotting:
		pylab.plot(x,un)
		pylab.show()
	
	if outputToFile:
		un.view(W)

print 'Done with', Nt, 'timesteps'

#print un.getArray()

#W = PETSc.Viewer().createASCII('test.txt')
#un.view(W)



