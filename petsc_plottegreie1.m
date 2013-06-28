tic; A = import_petsc_file('test.txt'); toc;

figure; pause(0.01)
min_ = min(min(A));
max_ = max(max(A));

for t=1:size(A,1)
    plot(A(t,:));
    axis([1 size(A,2) min_ max_]);
    title(t-1);
%     pause(0.0001);
    drawnow;
end