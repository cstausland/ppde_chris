# Import and initialize
import petsc4py, sys
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc

def differentiate_petsc(f_vec, a, b, n):
	x = np.linspace(a, b, n+1)
	dx = x[1] - x[0]
	df = np.zeros_like(x)       # df/dx
	
	f_first = PETSc.Vec().createMPI(f_vec.size-2,comm=PETSc.COMM_WORLD)
	f_first.setValues(range(f_vec.size-2), f_vec[:-2])
	
	f_last = PETSc.Vec().createMPI(f_vec.size-2,comm=PETSc.COMM_WORLD)
	f_last.setValues(range(f_vec.size-2), f_vec[2:])
	
	#f_gather = PETSc.Vec().createMPI(f_vec.size-2,comm=PETSc.COMM_WORLD)
	
	f_last.axpy(-1,f_first)	# Does y=alpha*x+y, and we set alpha=-1,
							# so we actually do y=y-x
	
	[start, stop] = f_last.getOwnershipRange()
	scat = PETSc.Scatter.toZero(f_last)
	scat.scatterBegin()
	scat.scatterEnd()
	
	#df[start+1:stop+1] = f_last.getArray()
	print PETSc.COMM_WORLD.getRank(), start, stop
	print PETSc.COMM_WORLD.getRank(), f_last.getArray()
	#df[1:-1] = f_last.getArray()
	df[1:-1] /= 2*dx
	
	df[0] = (f_vec[1]-f_vec[0])/dx
	df[-1] = (f_vec[-1]-f_vec[-2])/dx
	
	return df
	
	

def differentiate_vec(f_vec, a, b, n):
    '''
    Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference. Vectorized version.
    '''
    x = np.linspace(a, b, n+1)  # mesh
    df = np.zeros_like(x)       # df/dx
    dx = x[1] - x[0]
	
    # Internal mesh points
    df[1:-1] = (f_vec[2:] - f_vec[:-2])/(2*dx)
    # End points
    df[0]  = (f_vec[1]  - f_vec[0]) /dx
    df[-1] = (f_vec[-1] - f_vec[-2])/dx
    return df

# Start of script
if (len(sys.argv) > 1):
	n = int(sys.argv[1])
	if n == 1:
		print 'Error: n must be larger than 1'
		sys.exit()
else:
	n = 100
	
a=0; b=1;
x = np.linspace(a,b,n+1)
f = 3.*x**2

df_petsc = differentiate_petsc(f, a, b, n)
df_vec = differentiate_vec(f, a, b, n)

norm_difference = np.linalg.norm(df_petsc-df_vec)
if PETSc.COMM_WORLD.getRank() == 0:
	print 'Norm of the difference of the two df/dx solutions:'
	print norm_difference
	print 'If 0, the two solutions returned the same solution'