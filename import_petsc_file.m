function A = import_petsc_file(filen)
    fid = fopen(filen);
    A = [];
    line = 1;
    fgetl(fid); %fgetl(fid); fgetl(fid); 
    
    while 1
        l = fgetl(fid);
        if l == -1
            break;
        end
        
        %disp(l);
%         numb = sscanf(l,'%f');
        numb = sscanf(l,'%e');
        
        if isempty(numb)
            line=line+1;
            fgetl(fid);  % hopper over neste linje...
        else
%             disp(numb);
            A = [A numb];
        end
    end
    fclose(fid);
%     disp(line);
%     A = reshape(A,[size(A,2)/line line])';
    A = reshape(A,[size(A,2)/(line-1) line-1])';
end