# Kjoring:      python2.6 forward_1Dwave.py

# Import and initialize
import petsc4py, sys, numpy
petsc4py.init()
from petsc4py import PETSc

if (len(sys.argv) > 1):
	n = int(sys.argv[1])
else:
	n = 100



# Make temporary spacing under warning
print '\n\n\n'

Nt = 100
C2 = (2/1)**2
# Need our two matrices. One tridiag and one diag of size n-by-n     (should use sparse...)
tridiag = PETSc.Mat().createDense(n)
tridiag.setDiagonal(PETSc.Vec().createWithArray([2-2*C2]*n))
for i in range(n-1):
	tridiag.setValue(i+1,i,C2)
	tridiag.setValue(i,i+1,C2)

diag = PETSc.Mat().createDense(n)
diag.setDiagonal(PETSc.Vec().createWithArray([-1]*n))

# Create vectors, we need three different (one for each time steps)
#un = PETSc.Vec().createSeq.Duplicate(exp(numpy.exp(numpy.array(range(-n/2+1,n/2+1)))))
un = PETSc.Vec().createSeq(n)
unm1 = PETSc.Vec().createSeq(n)
#unm1 = un.duplicate()
#umn2 = PETSc.Vec().createSeq(n)
unm2 = un.duplicate()

un.setValues(range(n),numpy.exp(range(n)))
unm1.setValues(range(n),numpy.exp(range(n)))
unm2.setValues(range(n),numpy.exp(range(n)))

un.assemblyBegin()
un.assemblyEnd()
unm1.assemblyBegin()
unm1.assemblyEnd()
unm2.assemblyBegin()
unm2.assemblyEnd()
tridiag.assemblyBegin()
tridiag.assemblyEnd()
diag.assemblyBegin()
diag.assemblyEnd()

#for i in range(1,Nt):
	



print un.getArray()
print unm1.getArray()
print '\n\n'
print tridiag.getValues(range(n),range(n))

print 'Done'


