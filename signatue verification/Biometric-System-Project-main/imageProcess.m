function I = imageProcess(filename)
I=imread(filename);
I=imresize(I,[224,224]);
if ismatrix(I)
    I=cat(3,I,I,I);
end
end
