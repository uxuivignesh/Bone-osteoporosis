function im=Thresholding(I)
[r,c]=size(I);
im=zeros(r,c);
for i=1:r
    for j=1:c
        if I(i,j)>135
            im(i,j)=1;
        end
    end
end

end
