function [X] = normalizeL2(X)

X = bsxfun(@rdivide, X, sqrt(sum(X.*X,2)));