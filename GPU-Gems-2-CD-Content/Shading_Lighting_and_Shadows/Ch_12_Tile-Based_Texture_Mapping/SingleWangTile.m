%
% function [resultTile, sePath, enPath, nwPath, wsPath, pathCost] = ...
%       SingleWangTile(sTile, eTile, nTile, wTile, border_size)
%
% Inputs:
%
% sTile, eTile, nTile, wTile : 
%       the input tiles in south, east, north, and west
%       each tile must be sharp to the edge, 
%       for example, the bottom of sTile will be the s-edge of the result
%       input tiles should be slightly larger than the result to 
%       allow overlapping and cut optimization
% border_size: size of the tile border for control path center
%
% Return:
%
% resultTile: the result tile. 
%       height of the result tile == height of eTile/wTile
%       width of the result tile == width of sTile/nTile
%
% sePath, enPath, nwPath, wsPath: the 4 paths
%
% pathCost: total path cost
%
% Li-Yi Wei
% 8/20/2003
%

function [result, path_se, path_en, path_nw, path_ws, pathCost] = ...
    SingleWangTile(sTile, eTile, nTile, wTile, border_size)

% check arguments

if(nargin ~= 5)
  error('Wrong number of arguments');
end

if( (ndims(sTile) < 2) | (ndims(sTile) > 3) | ...
    (ndims(eTile) < 2) | (ndims(eTile) > 3) | ...
    (ndims(nTile) < 2) | (ndims(nTile) > 3) | ...
    (ndims(wTile) < 2) | (ndims(wTile) > 3) | ...
    (ndims(sTile) ~= ndims(eTile)) | ...
    (ndims(eTile) ~= ndims(nTile)) | ...
    (ndims(nTile) ~= ndims(wTile)) | ...
    (ndims(wTile) ~= ndims(sTile)) )
  error('Wrong dimensionality of the tiles');
end

if( (size(sTile, 2) ~= size(nTile, 2)) | ...
    (size(eTile, 1) ~= size(wTile, 1)) | ...
    (size(sTile, 1) <= size(eTile, 1)/2) | ...
    (size(sTile, 1) > size(eTile, 1)) | ...
    (size(eTile, 2) <= size(nTile, 2)/2) | ...
    (size(eTile, 2) > size(nTile, 2)) | ...
    (size(nTile, 1) <= size(wTile, 1)/2) | ...
    (size(nTile, 1) > size(wTile, 1)) | ...
    (size(wTile, 2) <= size(sTile, 2)/2) | ...
    (size(wTile, 2) > size(sTile, 2)) )
  error('Incompatible input tile sizes');
end
    
if( (size(sTile, 3) ~= size(eTile, 3)) | ...
    (size(eTile, 3) ~= size(nTile, 3)) | ...
    (size(nTile, 3) ~= size(wTile, 3)) | ...
    (size(wTile, 3) ~= size(sTile, 3)) )
  error('Incompatible input tile color spaces');
end

if(prod(size(border_size)) ~= 2)
  error('wrong border size');
end

% find out resultSize
resultSize = [size(eTile, 1) size(sTile, 2)];

% number of colors
numColors = size(sTile, 3);

% compute center
%resultCenter = ceil(resultSize/2);
if(0)
border_size = max(border_size, ...
    floor((resultSize - [size(sTile, 1) size(eTile, 2)])/2));
end

% s, e, n, w
borders = [border_size border_size];
borders = max(borders, [resultSize(1)-size(nTile, 1) resultSize(2)-size(wTile, 2) resultSize(1)-size(sTile, 1) resultSize(2)-size(eTile, 2)]+1);
resultCenter = [borders(3) borders(4)] + ...
    round(rand([1 2]) .* (resultSize - [borders(1)+borders(3) borders(2)+borders(4)]));

% find the 4 minimum cut paths
costSurface = ones(resultSize) * inf;
for k = 1:numColors
  costSurface_s(:, :, k) = costSurface;
  costSurface_e(:, :, k) = -costSurface;
  costSurface_n(:, :, k) = costSurface;
  costSurface_w(:, :, k) = -costSurface;
  costSurface_s(end-size(sTile, 1)+1:end, :, k) = sTile(:, :, k);
  costSurface_e(:, end-size(eTile, 2)+1:end, k) = eTile(:, :, k);
  costSurface_n(1:size(nTile, 1), :, k) = nTile(:, :, k);
  costSurface_w(:, 1:size(wTile, 2), k) = wTile(:, :, k);
end

% s and e
costSurface_all = costSurface_s - costSurface_e;
costSurface_all = costSurface_all .* costSurface_all;
costSurface = zeros(size(costSurface));
for k = 1:numColors
  costSurface = costSurface + costSurface_all(:, :, k);
end
[path_se, cost_se] = MinCostPath(costSurface, resultCenter, resultSize);

% e and n
costSurface_all = costSurface_e - costSurface_n;
costSurface_all = costSurface_all .* costSurface_all;
costSurface = zeros(size(costSurface));
for k = 1:numColors
  costSurface = costSurface + costSurface_all(:, :, k);
end
[path_en, cost_en] = MinCostPath(costSurface, [1 resultSize(2)], resultCenter);

% n and w
costSurface_all = costSurface_n - costSurface_w;
costSurface_all = costSurface_all .* costSurface_all;
costSurface = zeros(size(costSurface));
for k = 1:numColors
  costSurface = costSurface + costSurface_all(:, :, k);
end
[path_nw, cost_nw] = MinCostPath(costSurface, [1 1], resultCenter);

% w and s
costSurface_all = costSurface_w - costSurface_s;
costSurface_all = costSurface_all .* costSurface_all;
costSurface = zeros(size(costSurface));
for k = 1:numColors
  costSurface = costSurface + costSurface_all(:, :, k);
end
[path_ws, cost_ws] = MinCostPath(costSurface, resultCenter, [resultSize(1) 1]);

% fill the result tile according to the 4 paths found above
result = zeros([resultSize numColors]);

for k = 1:numColors
  % s and e quadrant
  for n = 1:size(path_se, 1)
    result(path_se(n, 1), resultCenter(2):path_se(n, 2), k) = ...
        costSurface_s(path_se(n, 1), resultCenter(2):path_se(n, 2), k);
    result(path_se(n, 1), path_se(n, 2):end, k) = ...
        costSurface_e(path_se(n, 1), path_se(n, 2):end, k);
  end
  
  % e and n quadrant
  for n = 1:size(path_en, 1)
    result(path_en(n, 1), resultCenter(2):path_en(n, 2), k) = ...
        costSurface_n(path_en(n, 1), resultCenter(2):path_en(n, 2), k);
    result(path_en(n, 1), path_en(n, 2):end, k) = ...
        costSurface_e(path_en(n, 1), path_en(n, 2):end, k);
  end

  % n and w quadrant
  for n = 1:size(path_nw, 1)
    result(path_nw(n, 1), 1:path_nw(n, 2), k) = ...
        costSurface_w(path_nw(n, 1), 1:path_nw(n, 2), k);
    result(path_nw(n, 1), path_nw(n, 2):resultCenter(2), k) = ...
        costSurface_n(path_nw(n, 1), path_nw(n, 2):resultCenter(2), k);
  end

  % w and s quadrant
  for n = 1:size(path_ws, 1)
    result(path_ws(n, 1), 1:path_ws(n, 2), k) = ...
        costSurface_w(path_ws(n, 1), 1:path_ws(n, 2), k);
    result(path_ws(n, 1), path_ws(n, 2):resultCenter(2), k) = ...
        costSurface_s(path_ws(n, 1), path_ws(n, 2):resultCenter(2), k);
  end
end

pathCost = cost_se/size(path_se, 1) + cost_en/size(path_en, 1) + cost_nw/size(path_nw, 1) + cost_ws/size(path_ws, 1);

% done
return;
