fprintf('Loading dataset...\n');
load('celebrity2000');

fprintf('Computing PCA feature...\n');
eps = 10^-5;
nPart = 16;
pcaDim = 500;
partDim = 4720;
cPts = size(celebrityImageData.identity,1);
celebrityImageData.pcaFeature = zeros(cPts, pcaDim*nPart);
changeIndex = reshape([1:75520], [], 5)';
changeIndex = changeIndex(:);
for p = 1:nPart
   partIndex = changeIndex([1 + (p-1)*partDim:p*partDim]);
   pcaIndex = [1 + (p-1)*pcaDim:p*pcaDim];
   X = sqrt(double(celebrityImageData.feature(:,partIndex)));
   [~, PCAmapping] = pca(X(find(celebrityImageData.rank > 35), :), pcaDim);
   X_PCA = bsxfun(@minus, X, PCAmapping.mean) * PCAmapping.M;
   W = diag(ones(pcaDim,1)./sqrt(PCAmapping.lambda + eps));
   X_PCA = X_PCA*W;
   celebrityImageData.pcaFeature(:,pcaIndex) = X_PCA;
end

fprintf('Computing Cross-Age Reference Coding...\n');
databaseIndex{1} = find((celebrityImageData.year == 2004 | celebrityImageData.year == 2005 | celebrityImageData.year == 2006) & celebrityImageData.rank <=5 & celebrityImageData.rank > 2);
databaseIndex{2} = find((celebrityImageData.year == 2007 | celebrityImageData.year == 2008 | celebrityImageData.year == 2009) & celebrityImageData.rank <=5 & celebrityImageData.rank > 2);
databaseIndex{3} = find((celebrityImageData.year == 2010 | celebrityImageData.year == 2011 | celebrityImageData.year == 2012) & celebrityImageData.rank <=5 & celebrityImageData.rank > 2);
queryIndex = find(celebrityImageData.year == 2013 & celebrityImageData.rank <=5 & celebrityImageData.rank > 2);
lambda = 10;
lambda2 = 10000;
CARC_query = CARC(celebrityImageData, celebrityData, lambda, lambda2, queryIndex);
CARC_database{1} = CARC(celebrityImageData, celebrityData, lambda, lambda2, databaseIndex{1});
CARC_database{2} = CARC(celebrityImageData, celebrityData, lambda, lambda2, databaseIndex{2});
CARC_database{3} = CARC(celebrityImageData, celebrityData, lambda, lambda2, databaseIndex{3});
dataset{1} = '2004-2006';
dataset{2} = '2007-2009';
dataset{3} = '2010-2012';


%Here is for prepareing your own features, the order of the features should be same as "image.list"
%{
celebrityImageData.newFeature = zeros(163446, feature_dim);
%}

fprintf('Evaluation...\n');
queryId = celebrityImageData.identity(queryIndex);
for i = 1:3
   fprintf(['Result for dataset ' dataset{i} '\n']);
   databaseId = celebrityImageData.identity(databaseIndex{i});
   qX = celebrityImageData.pcaFeature(queryIndex, :);
   X = celebrityImageData.pcaFeature(databaseIndex{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';
   result = evaluation(dist, queryId, databaseId);
   fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   dist = -1*normalizeL2(CARC_query)*normalizeL2(CARC_database{i})';
   result = evaluation(dist, queryId, databaseId);
   fprintf('CARC:\t\t\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   
   %To compare your own features using same protocol, you can uncomment these lines for evaluation
   %{
   qX = celebrityImageData.newFeature(queryIndex, :);
   X = celebrityImageData.newFeature(databaseIndex{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';
   result = evaluation(dist, queryId, databaseId);
   fprintf('New Features:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   %}
end
