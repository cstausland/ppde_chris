nx = 51;
ny = 51;
Nt = 25; Nt=Nt+1;

tic; A = import_petsc_file_2d('test2d_7.txt',nx,ny,Nt); toc;
A = permute(A,[2 1 3]);

figure; pause(2);
axis([1 nx 1 ny]);
min_ = min(min(min(A)));
max_ = max(max(max(A)));

for t=1:Nt
    surf(A(:,:,t));
    axis([1 nx 1 ny min_ max_]);
    caxis([min_ max_]);
    title(t-1);
    pause(0.5);
end