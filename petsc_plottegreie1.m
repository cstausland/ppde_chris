clear; tic; A = import_petsc_file('test.txt'); toc;

figure; pause(0.01)
min_ = min(min(A));
max_ = max(max(A));

for t=1:size(A,1)
    plot(A(t,:));
    axis([1 size(A,2) min_ max_]);
    title([num2str(t-1) ' of ' num2str(size(A,1)-1)]);
%     pause(0.001);
    drawnow;
end