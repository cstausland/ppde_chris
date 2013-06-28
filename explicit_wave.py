# Import and initialize
import petsc4py, sys, numpy
petsc4py.init(sys.argv)
from petsc4py import PETSc

if (len(sys.argv) > 1):
	n = int(sys.argv[1])
else:
	n = 100

# Temporal points (number of time steps)
if (len(sys.argv) > 2):
	Nt = int(sys.argv[2])
else:
	Nt = 100


a_from = 0; b_to = 1; dx = (b_to-a_from)/(n-1.); C = 0.001; dt = numpy.sqrt(C)*dx
I = numpy.exp(-numpy.square(numpy.linspace(a_from,b_to,n)-0.5*(b_to-a_from))*50)

T = PETSc.Mat().createAIJ([n, n],nnz=3)
for i in range(1, n):   # filling the diagonal and off-diagonal entries
	T.setValue(i,i,2.-2*C)
	T.setValue(i-1,i,C)
	T.setValue(i,i-1,C)
T.setValue(0,0,1); T.setValue(0,1,0);
T.setValue(n-1,n-1,1); T.setValue(n-1,n-2,0);

unp1 = PETSc.Vec().createSeq(n)
un = PETSc.Vec().createSeq(n)
unm1 = PETSc.Vec().createSeq(n)

un.setValues(range(n),I)
un.setValue(0,0); un.setValue(n-1,0);
unm1.setValues(range(n),I)
unm1.setValue(0,0); unm1.setValue(n-1,0);

T.assemblyBegin(); T.assemblyEnd();
unp1.assemblyBegin(); unp1.assemblyEnd();
un.assemblyBegin(); un.assemblyEnd();
unm1.assemblyBegin(); unm1.assemblyEnd();

outputToFile = True
if outputToFile:
	W = PETSc.Viewer().createASCII('test.txt',format=1)
	unm1.view(W)
	un.view(W)

for t in range(Nt):
	# We want to subtract the vector, that is, add -1*vec
	unm1.scale(-1)
	T.multAdd(un, unm1, unp1)
	#T.multAdd(unp1, unm1, un)
	#T.multAdd(un, unm1, unp1)
	
	un.copy(unm1)
	unp1.copy(un)
	
	un.setValue(0,0); un.setValue(n-1,0);
	
	if outputToFile:
		unp1.view(W)

if PETSc.COMM_WORLD.getRank() == 0:
	print 'Done with', Nt, 'timesteps'
#