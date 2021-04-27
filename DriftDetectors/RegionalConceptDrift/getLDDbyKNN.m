function ldd = getLDDbyKNN(insts1, instsBase, knn)
    
    numBase = size(instsBase, 1);
    nanb = knnsearch([instsBase; insts1], instsBase, 'K', knn);
    na = sum(nanb<=numBase,2);
    nb = sum(nanb>numBase,2);
    ldd = nb./size(insts1, 1) ./ (na./size(instsBase, 1)) -1;
    
end