nx = 25;
ny = 25;
Nt = 1000; Nt=Nt+1;

tic; A = import_petsc_file_2d('test2d.txt',nx,ny,Nt); toc;

figure; pause(2);
min_ = min(min(min(A)));
max_ = max(max(max(A)));

for t=1:Nt
    surf(A(:,:,t));
    axis([1 nx 1 ny min_ max_]);
    caxis([min_ max_]);
    title(t-1);
    pause(0.001);
end