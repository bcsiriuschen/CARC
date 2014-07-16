% Calculate mean average precision
function [result] = evaluation(dist, queryId, databaseId)
   qPts = size(queryId,1);
   nPts = size(databaseId,1);
   totalK = 20;
   ap = zeros(qPts, 1);
   patK = zeros(qPts, totalK);
   rankResults = zeros(qPts, nPts);
   for i = 1:qPts
      [~, idx] = sort(dist(i,:), 'ascend');
      correctRank = find(databaseId(idx) == queryId(i));
      rankResults(i,:) = idx;
      nAns = size(correctRank,1);
      for j = 1:nAns
         ap(i) = ap(i) + j/correctRank(j);
      end
      for k = 1:totalK
         patK(i,k) = sum(databaseId(idx(1:k)) == queryId(i))/k;
      end
      ap(i) = ap(i)/nAns;
   end
   result.ap = ap;
   result.patK = mean(patK);
   result.rankResults = rankResults;