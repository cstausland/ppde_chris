# Kjoring:      python2.6 backward_wave_solver.py

# Import and initialize
import petsc4py, sys, numpy
petsc4py.init(sys.argv)   # kan man legge til input her for aa velge mellom mpich og OpenMPI?
from petsc4py import PETSc

if (len(sys.argv) > 1):
	n = int(sys.argv[1])
else:
	n = 100



# Make temporary spacing under warning
print '\n\n\n'

Nt = 100   # Number of timesteps
C2 = (2/1)**2    # C = dt/dx

### These are big matrices and vectors, so we must eventually use sparse types.
### Using dense for now. Use small n

# Need our two matrices. One tridiag and one diag of size n-by-n
tridiag = PETSc.Mat().createDense(n)
tridiag.setDiagonal(PETSc.Vec().createWithArray([1+2*C2]*n))
for i in range(n-1):   # filling the off-diagonal entries
	tridiag.setValue(i+1,i,-C2)
	tridiag.setValue(i,i+1,-C2)

diagm1 = PETSc.Mat().createDense(n)
diagm1.setDiagonal(PETSc.Vec().createWithArray([-1]*n))

diag2 = PETSc.Mat().createDense(n)
diag2.setDiagonal(PETSc.Vec().createWithArray([2]*n))

# Create vectors, we need three different (one for each time steps)
un = PETSc.Vec().createSeq(n)
un.set(1)
unm1 = PETSc.Vec().createSeq(n)
#unm1 = un.duplicate()
#umn2 = PETSc.Vec().createSeq(n)
unm2 = un.duplicate()
b = PETSc.Vec().createSeq(n)

# Setting init value (exp-function around x=0) and using
# that du/dt=0 at t=0 (which makes umn1 = umn2)
unm1.setValues(range(n),numpy.exp(numpy.divide(range(-n/2+1,n/2+1),100.)))
unm2.setValues(range(n),numpy.exp(numpy.divide(range(-n/2+1,n/2+1),100.)))

# Assemble all vectors and matrices (is this really necessary for all the vectors?)
un.assemblyBegin()
un.assemblyEnd()
unm1.assemblyBegin()
unm1.assemblyEnd()
unm2.assemblyBegin()
unm2.assemblyEnd()
tridiag.assemblyBegin()
tridiag.assemblyEnd()
diagm1.assemblyBegin()
diagm1.assemblyEnd()
diag2.assemblyBegin()
diag2.assemblyEnd()



# Calculate RHS (b): A*un = b = diag2*unm1 + diagm1*unm2
diag2.mult(unm1,b)
diagm1.multAdd(unm2,b,b)

# Solve A*x = b one time using a direct solver.
# Should also implement a solver using conjugent gradient or
# similar Krylov methods (iterativ methods, versus direct
# methods as Gaussian elimination and LU prec.)
'''tridiag.solve(b,un)'''


#tridiag.mult(b,un)


# Example from:
#https://www.tacc.utexas.edu/c/document_library/get_file?uuid=c2de5a2a-8e08-4f6e-8d49-2086681520d1&groupId=13601
# Conjugate gradient solver. Something like this:

ksp = PETSc.KSP()
ksp.create()
ksp.setType('cg')   # conjugate gradient
#ksp.getPC().setType('icc')   # incomplete Cholesky
ksp.setOperators(tridiag)
#ksp.setFromOptions()   # uncomment if options were set during init(?, see ksp_serial.py, line 49)
ksp.solve(b,un)



print '\n\n\n'
print un.getArray()
print unm1.getArray()
print unm2.getArray()
print '\n\n' 
#tridiag.view()    # <--- sykt stygg formattering
print tridiag.getValues(range(n),range(n)), '\n'
print diag2.getValues(range(n),range(n)), '\n'
print diagm1.getValues(range(n),range(n)), '\n'

print 'Done'


