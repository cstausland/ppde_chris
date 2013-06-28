function A = import_petsc_file_2d(filen,nx,ny,Nt)
    fid = fopen(filen);
    A = NaN(nx,ny,Nt);
    Atemp = NaN(1,nx*ny);
    
    for k=1:Nt
        fgetl(fid); fgetl(fid); fgetl(fid);
        for i=1:nx*ny
            l = fgetl(fid);
            numb = sscanf(l,'%f');
            
            Atemp(i) = numb;
        end
        
        A(:,:,k) = reshape(Atemp,[nx ny]);
    end
%     while 1
%         l = fgetl(fid);
%         if l == -1
%             break;
%         end
%         
%         numb = sscanf(l,'%f');
%         A = [A numb];
%     end


    fclose(fid);
%     A = Atemp;
    %A = reshape(A,[nx ny]);
end