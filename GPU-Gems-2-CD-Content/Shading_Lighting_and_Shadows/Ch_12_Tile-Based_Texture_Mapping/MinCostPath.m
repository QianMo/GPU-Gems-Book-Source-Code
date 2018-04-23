%
% function [resultPath, pathCost] = ...
%    MinCostPath(errorSurface, startPoint, endPoint)
%
% Inputs:
%
% errorSurface: the 2D scalar error surface
% startPoint, endPoint: the (row, col) positions of start and end points
%
% Return:
%
% resultPath:
% the minimum cost path in matrix of size Nx2, 
% where N are the points on the path including start/end points
%
% pathCost: the cost of the path
%
% Li-Yi Wei
% 8/19/2003
%

function [result, resultCost] = ...
    MinCostPath(errorSurface, startPoint, endPoint)

% check arguments
if(nargin ~= 3)
  error('Wrong number of input arguments');
end

if(ndims(errorSurface) ~= 2)
  error('error surface must be 2D');
end

if( (prod(size(startPoint)) ~= 2) | (prod(size(endPoint)) ~= 2) )
  error('start and end points must be 2D');
end

if(any(startPoint <= 0) | any(startPoint > size(errorSurface)) | ...
   any(endPoint <= 0) | any(endPoint > size(errorSurface)) )
  error('illegal start or end points');
end

errorSurface(isnan(errorSurface)) = inf;
if(min(errorSurface(:)) == inf)
  error('error surface is inf');
end

if(all(startPoint == endPoint))
  result = startPoint;
  return;
end

% the algorithm proceeds vertically, 
% so transpose everything if endPoint - startPoint is wider than taller
transposed = 0;
if(abs(endPoint(1) - startPoint(1)) < abs(endPoint(2) - startPoint(2)))
  errorSurface = errorSurface';
  
  temp = startPoint(1); 
  startPoint(1) = startPoint(2); 
  startPoint(2) = temp;
  
  temp = endPoint(1); 
  endPoint(1) = endPoint(2); 
  endPoint(2) = temp;
  
  transposed = 1;
end

% make sure start point is on top and end point is on bottom
flip_top_down = 0;
if(endPoint(1) < startPoint(1))
  temp = endPoint; endPoint = startPoint; startPoint = temp;
  flip_top_down = 1;
end

% make sure start point is on left and end point is on right
flip_left_right = 0;
flip_left_right_value = size(errorSurface, 2)+1;
if(endPoint(2) < startPoint(2))
  newErrorSurface = errorSurface;
  for k = 1:size(newErrorSurface, 2)
    newErrorSurface(:, k) = errorSurface(:, flip_left_right_value-k);
  end
  errorSurface = newErrorSurface;
  startPoint(2) = flip_left_right_value - startPoint(2);
  endPoint(2) = flip_left_right_value - endPoint(2);
  flip_left_right = 1;
end

% rotate 45 degrees if startPoint to endPoint is closer to diagonal
rot45 = 0;

rot45MaxSize = max(size(errorSurface))*2-1;
rot45Middle = ceil(rot45MaxSize/2);
  
if(abs(startPoint(1)-endPoint(1)) < 2*abs(startPoint(2)-endPoint(2)))
  newErrorSurface = ones([rot45MaxSize rot45MaxSize])*inf;
  for row = 1:size(errorSurface, 1)
    for col = 1:size(errorSurface, 2)
      newErrorSurface(row+col-1, col-row+rot45Middle) = ...
          errorSurface(row, col);
    end
  end
  
  newStartPoint = ...
      [startPoint(1)+startPoint(2)-1, ...
        startPoint(2)-startPoint(1)+rot45Middle];
  newEndPoint = ...
      [endPoint(1)+endPoint(2)-1, ...
        endPoint(2)-endPoint(1)+rot45Middle];
  
  errorSurface = newErrorSurface;
  startPoint = newStartPoint;
  endPoint = newEndPoint;
  
  rot45 = 1;
end

% truncate the error surface
errorSurface = errorSurface(startPoint(1):endPoint(1), :);

row_offset = startPoint(1)-1;

startPoint(1) = startPoint(1) - row_offset;
endPoint(1) = endPoint(1) - row_offset;

% setup the cost surface
costSurface = ones(size(errorSurface))*inf;

costSurface(endPoint(1), endPoint(2)) = 0;

% setup the link surface
linkSurface = ones([size(errorSurface) 2])*inf;

% setup the result
result = zeros(size(costSurface, 1), 2);

result(1, :) = startPoint;
result(end, :) = endPoint;

% compute the cost surface and link surfaces
for row = size(costSurface, 1)-1:-1:1
  for col = 1:size(costSurface, 2)
    
    col_next_start = col-1; col_next_end = col+1;
    if(col_next_start <= 0) 
      col_next_start = 1; 
    end
    if(col_next_end >= size(costSurface, 2)) 
      col_next_end = size(costSurface, 2); 
    end
      
    for col_next = col_next_start:col_next_end
      cost_now = ...
          costSurface(row+1, col_next) + errorSurface(row+1, col_next);
      
      if(cost_now < costSurface(row, col))
        costSurface(row, col) = cost_now;
        linkSurface(row, col, 1) = row+1;
        linkSurface(row, col, 2) = col_next;
      end
    end
  end
end

% compute the result from cost and link surfaces
resultCost = costSurface(result(1, 1), result(1, 2));

for k = 2:size(result, 1)
  result(k, 1) = linkSurface(result(k-1, 1), result(k-1, 2), 1);
  result(k, 2) = linkSurface(result(k-1, 1), result(k-1, 2), 2);
  if(any(result(k, :) == inf))
    error('inf is blocking the path');
  end
  resultCost = resultCost + costSurface(result(k, 1), result(k, 2));
end

% offset back

for k=1:size(result, 1)
  result(k, 1) = result(k, 1) + row_offset;
end

% rot45 back

if(rot45)
  newResult = result;
  
  for k=1:size(newResult, 1)
    newResult(k, :) = ...
        [result(k, 1)-result(k, 2)+rot45Middle+1, ...
          result(k, 1)+result(k, 2)-rot45Middle+1]/2;
  end
  
  result = newResult;
  
  rot45 = 0;
end

% flip_left_right back
if(flip_left_right)
  
  for k = 1:size(result, 1)
    result(k, 2) = flip_left_right_value - result(k, 2);
  end
  
  flip_left_right = 0;
end

% flip_top_down back
if(flip_top_down)
  newResult = result;
  
  for k=1:size(newResult, 1)
    newResult(k, :) = result(size(newResult, 1) - k + 1, :);
  end
  
  result = newResult;
  
  flip_top_down = 0;
end

% transpose back
if(transposed)
  newResult = result;
  
  newResult(:, 1) = result(:, 2);
  newResult(:, 2) = result(:, 1);
  
  result = newResult;
  
  transposed = 0;
end

return;
