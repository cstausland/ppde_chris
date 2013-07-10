# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc

'''
CHANGE THIS TO USE A FUNCTION (NOT A MATRIX-MULTIPLICATION) TO SOLVE FORWARD. USE globalToLocal and localToGlobal
BETWEEN THEM FOR COMMUNICATION. REMEMBER TO EXPLOIT GHOST POINTS (well, it rather let us NOT worry about that...)
'''


def solver_FE_PETSc(I, a, L, Nx, C, T):
	# Change this to NOT use mat-mult...
	# Verify using timings (with both a numpy-solver and Cython-solver)
	t0 = time.clock()
	x = np.linspace(0, L, Nx+1)
	dx = x[1] - x[0]
	dt = C*dx**2/a
	Nt = int(round(T/float(dt)))
	t = np.linspace(0, T, Nt+1)
	
	# Does DA contribute when using matrix solvers?
	# Boundary-type is mirrored, stencil-type 2 is star (we dont need box)
	da = PETSc.DA().create(sizes=[Nx+1], boundary_type=2, stencil_type=0, stencil_width=1)
	da.setUniformCoordinates(0, L)   # We never really use this in here
	#print da.getCoordinates().getArray()   # Might use this when setting functions that needs
											# coordinates, this makes a linspace kind of array
	u = da.createGlobalVector(); u_1 = da.createGlobalVector();
	A = da.createMatrix(); A.setType('aij')  # type looks like seqaij as default, must change this
	
	# Initialize init time step
	u_1.setValues(range(Nx+1),I)
	
	# Initialize the matrix A. Could maybe get this through DA, but do we want want that?
	# This means one creation call and one allocation call? What positive properties do we get?
	A = PETSc.Mat().createAIJ([Nx+1, Nx+1],nnz=3)
	for i in range(1, Nx+1):   # filling the non-zero entires
		A.setValue(i,i,1.-2*C)
		A.setValue(i-1,i,C)
		A.setValue(i,i-1,C)
	A.setValue(0,0,1); A.setValue(0,1,0);
	A.setValue(Nx,Nx,1); A.setValue(Nx,Nx-1,0);
	
	local_vec = da.createLocalVector(); local_vec_new = da.createLocalVector();
	# Assemble the vectors and the matrix. This distribute the objects out over the procs.
	A.assemble(); u.assemble(); u_1.assemble();
	
	# This can be cut out, this is for plotting in external
	# programs
	if PETSc.Options().getBool('toFile',default=False):
		if PETSc.COMM_WORLD.getSize() > 1:
			if PETSc.COMM_WORLD.getRank() == 0:
				print 'Warning: NOT writing to file (using parallel)'
				PETSc.Options().setValue('toFile',False)
		else:
			W = PETSc.Viewer().createASCII('test3.txt',format=1);
			u_1.view(W)
	
	
	for n in range(0, Nt):
		da.globalToLocal(u_1, local_vec)
		#local_vec_new = advance_FE_looper(local_vec, C, da)
		local_vec_new = advance_FE_fastnumpy(local_vec, C, da)
		da.localToGlobal(local_vec_new, u_1)
	# Maa plotte for aa bekrefte at dette ble riktig... Kan jo hende det bare ble surr
	
		if PETSc.Options().getBool('draw',default=False):
			U = da.createNaturalVector()
			da.globalToNatural(u_1,U)
			scatter, U0 = PETSc.Scatter.toZero(U)
			scatter.scatter(U,U0,False,PETSc.Scatter.Mode.FORWARD)
			rank = PETSc.COMM_WORLD.getRank()
			solution = U.copy()
			draw = PETSc.Viewer.DRAW()
			draw(solution)
			time.sleep(0.001)
			#time.sleep(10)
	
	return time.clock()-t0
#

def advance_FE_looper(u_1, C, da):
	u = da.createLocalVector()
	u_1.duplicate(u)
	for i in range(1,u_1.getSize()-1):
		val = C*u_1.getValue(i-1) + (1.-2*C)*u_1.getValue(i) + C*u_1.getValue(i+1)
		u.setValue(i, val)
	return u

def advance_FE_fastnumpy(u_1, C, da):
	u = u_1.getArray()   # getArray gjoer ingen hardcopy
	# Do work
	u_advance = np.zeros_like(u)
	u_advance[1:-1] = C*u[:-2] + (1.-2*C)*u[1:-1] + C*u[2:]
	
	# createWithArray bruker samme memory space, og gjoer ikke en hardcopy.
	u_return = PETSc.Vec().createWithArray(u_advance)
	return u_return

# Dette kan jo settes rett i funksjonen ogsaa, og ikke hentes fra funksjonskallet...
a = PETSc.Options().getReal('a',default=1.0)
L = PETSc.Options().getReal('L',default=1.0)
Nx = PETSc.Options().getInt('Nx',default=100)
C = PETSc.Options().getReal('C',default=0.5)
T = PETSc.Options().getReal('T',default=0.25)

x = np.linspace(0,L,Nx+1)
I = np.exp(-np.square(x-0.5*L)*50)

t_time_fe = solver_FE_PETSc(I, a, L, Nx, C, T)

#if PETSc.COMM_WORLD.getRank() == 0:
#	print 'Timings:\n   FE: ', t_time_fe