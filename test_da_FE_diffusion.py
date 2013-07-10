# Import and initialize
import petsc4py, sys, time
import numpy as np
petsc4py.init(sys.argv)
from petsc4py import PETSc

'''
Kan proeve aa lage en eksplisitt loeser av enten wave eller diffusjon hvor man ikke bruker matriser, men
bruker globaltolocal, loeser local enten med python-funksjon (kan proeve det for enkelhets skyld foerst)
eller cython (senere om det fungerer bra, for hastighet), og husk at det er ghost points inkludert da,
og saa kjoere en localtoglobal etterpaa (for aa oppdatere). Det blir matrisefritt saann som HPL ville ha.
'''

def solver_FE_PETSc(I, a, L, Nx, C, T):
	
	t0 = time.clock()
	x = np.linspace(0, L, Nx+1)
	dx = x[1] - x[0]
	dt = C*dx**2/a
	Nt = int(round(T/float(dt)))
	t = np.linspace(0, T, Nt+1)
	
	# Does DA contribute when using matrix solvers?
	da = PETSc.DA().create(sizes=[Nx+1], boundary_type=2, stencil_type=0, stencil_width=1)
	da.setUniformCoordinates(0, L)   # We never really use this in here
	#print da.getCoordinates().getArray()   # Might use this when setting functions that needs
											# coordinates
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
		# Solve for next step. This should be done without
		# matrix multiplication (solving the system directly).
		A.mult(u_1, u)
		
		# Copy new solution to the old vector
		u.copy(u_1)
		
		
		# Set boundary condition
		u_1.setValue(0,0); u_1.setValue(Nx,0)
		
		if PETSc.Options().getBool('toFile',default=False):
			u_1.view(W)
	
	# Drawing works somehow with parallel (maybe it lets root rank does the job?),
	# though PETSc's drawing capabilities are limited. Only for simple verifications...
		if PETSc.Options().getBool('draw',default=False):
			U = da.createNaturalVector()
			da.globalToNatural(u_1,U)
			scatter, U0 = PETSc.Scatter.toZero(U)
			scatter.scatter(U,U0,False,PETSc.Scatter.Mode.FORWARD)
			rank = PETSc.COMM_WORLD.getRank()
			solution = U.copy()
			draw = PETSc.Viewer.DRAW()
			draw(solution)
			#time.sleep(10)
	return time.clock()-t0
#



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