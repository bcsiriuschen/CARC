% Cross-Age Celebrity Coding
% Reference: Bor-Chun Chen, Chu-Song Chen, Winston Hsu. Cross-Age Reference Coding for Age-Invariant Face Recognition and Retrieval, ECCV, 2014
% http://bcsiriuschen.github.io/CARC/
function [CRAC_Feature] = CARC(celebrityImageData, celebrityData, lambda, lambda2, imageIndex)

%initialize some variables
nPart = 16;
pcaDim = 500;
cNum = size(find(celebrityData.rank <= 20 & celebrityData.rank > 5),1);
nPts = size(imageIndex,1);
CRAC_Feature = zeros(nPts, cNum*nPart);
celebrityIdentity = find(celebrityData.rank <= 20 & celebrityData.rank > 5);
groupNum = 10;
group{groupNum,cNum} = 0;
for i = 1:cNum
   group{1,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2004);
   group{2,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2005);
   group{3,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2006);
   group{4,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2007);
   group{5,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2008);
   group{6,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2009);
   group{7,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2010);
   group{8,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2011);
   group{9,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2012);
   group{10,i} = find(celebrityImageData.identity == celebrityIdentity(i) & celebrityImageData.year == 2013);
end
L = zeros(cNum*(groupNum-2), cNum*groupNum);
for j = 1:(groupNum-2)
   L(1 + (j-1)*cNum:j*cNum, 1 + (j-1)*cNum:j*cNum) = eye(cNum);
   L(1 + (j-1)*cNum:j*cNum, 1 + j*cNum:(j+1)*cNum) = -2*eye(cNum);
   L(1 + (j-1)*cNum:j*cNum, 1 + (j+1)*cNum:(j+2)*cNum) = eye(cNum);
end

for p = 1:nPart
   partIndex = [1 + (p-1)*pcaDim:p*pcaDim];
   partAll = celebrityImageData.pcaFeature(:,partIndex);
   partAll = normalizeL2(partAll);
   partX = partAll(imageIndex,:);
   partX = repmat(partX, 1, groupNum);
   partD = zeros(groupNum*pcaDim, groupNum*cNum);
   
   %Step-1 Reference Set Representations
   for j = 1:groupNum
      cX = zeros(cNum, pcaDim);
      for i = 1:cNum
         if(size(group{j,i},1) ~= 0)
            cX(i,:) = mean(partAll(group{j,i},:));
         end
      end
      partD(1+(j-1)*pcaDim:j*pcaDim, 1+(j-1)*cNum:j*cNum) = cX';
   end

   %Step-2 Encoding Feature
   A = (partD'*partD + lambda*eye(size(partD,2)) + lambda2*L'*L)\partD'*partX';

   %Step-3 Max Pooling
   A = reshape(A, cNum, groupNum, nPts);
   result = zeros(cNum, nPts);
   resultSign = zeros(cNum, nPts);
   result(:,:) = max(abs(A), [], 2);
   resultSign(:,:) = max(A, [], 2);
   resultSign = double(resultSign == result);
   resultSign(find(resultSign == 0)) = -1;
   result = resultSign.*result;
   CRAC_Feature(:,1+(p-1)*cNum:p*cNum) = result';
end
