%
% function [result] = WangTileIndex(startEdge, endEdge)
%
% Input:
% 
% startEdge, endEdge : 
%   the numbering of start and end edges (starting from 1)
%   
% Return:
%
% result :
%   the 1D sequential order of this edge combination (starting from 1)
%
% Li-Yi Wei
% 8/21/2003
%

function [result] = WangTileIndex(startEdge, endEdge)

% check arguments

if(nargin ~= 2)
  error('Wrong number of arguments');
end

startEdge = round(startEdge); endEdge = round(endEdge);

if( (startEdge < 1) | (endEdge < 1) )
  error('wrong startEdge or endEdge');
end

x = startEdge - 1; y = endEdge - 1;

result = -1;

if(x < y)
  result = (2*x + y*y);
elseif(x == y)
  if(x > 0)
    result = ((x+1)*(x+1) - 2);
  else
    result = 0;
  end
else 
  if(y > 0)
    result = (x*x + 2*y - 1);
  else
    result = ((x+1)*(x+1) - 1);
  end
end

result = result + 1;

% done
return;

