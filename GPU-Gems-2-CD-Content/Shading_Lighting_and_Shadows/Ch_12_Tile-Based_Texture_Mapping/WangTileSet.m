%
% 
% function [resultTiles, tileError] = ...
%   WangTileSet(sampleTexture, tileColors, tileSize, borderSize)
%
% Inputs:
%
% sampleTexture : 
%   the sample texture to build tiles from
% 
% tileColors : 
%   [numVerticalColors numHorizontalColors numTilesPerColor]
%
% tileSize :
%   [height width] of each tile
%
% borderSize :
%   see SingleWangTile
%
% Return:
%
% resultTiles :
%   the result set of tiles packed into one big image
%   the packing is determined by WangTileIndex
%
% tileError :
%   the quilting errors of resultTiles
%
% Li-Yi Wei
% 8/21/2003
%

function [resultTiles, tileError] = ...
    WangTileSet(sampleTexture, tileColors, tileSize, borderSize)

% constants
tileMargin = 2;

% check arguments
if(nargin ~= 4)
  error('Wrong number of input arguments');
end

if((ndims(sampleTexture) < 2) | (ndims(sampleTexture) > 3))
  error('Wrong sample texture dimension');
end

if(prod(size(tileColors)) ~= 3)
  error('Wrong tileColors specification');
end

if(prod(size(tileSize)) ~= 2)
  error('Wrong tileSize specification');
end

if(prod(size(borderSize)) ~= 2)
  error('Wrong borderSize specification');
end

[sampleHeight sampleWidth sampleDepth] = size(sampleTexture);

if(any([sampleHeight sampleWidth] <= (tileSize + 2*[tileMargin tileMargin])))
  error('sample texture smaller than tile size');
end

numVColors = tileColors(1);
numHColors = tileColors(2);
numTilesPerColor = tileColors(3);

if(numTilesPerColor ~= 1)
  warning('can only handle 1 tile per color combination');
end

% choose sample tile locations and sizes
vTileSamples = zeros([numVColors 4]); % for east and west
hTileSamples = zeros([numHColors 4]); % for north and south

tileHeight = tileSize(1); tileWidth = tileSize(2);

for k = 1:size(vTileSamples, 1)
  vTileSamples(k, 1:2) = ...
      rand([1 2]) .* ...
      [sampleHeight-tileHeight sampleWidth-tileWidth-tileMargin];
  vTileSamples(k, 1:2) = ceil(vTileSamples(k, 1:2));
  vTileSamples(k, 3) = tileHeight;
  vTileSamples(k, 4) = sampleWidth-vTileSamples(k, 2)+1;
  if(vTileSamples(k, 4) > 2*tileWidth)
    vTileSamples(k, 4) = 2*tileWidth;
  end

  vTileSamples(k, 3:4) = ...
      vTileSamples(k, 3:4) + vTileSamples(k, 1:2) - 1;
 
end

for k = 1:size(hTileSamples, 1)
  hTileSamples(k, 1:2) = ...
      rand([1 2]) .* ...
      [sampleHeight-tileHeight-tileMargin sampleWidth-tileWidth];
  hTileSamples(k, 1:2) = ceil(hTileSamples(k, 1:2));
  hTileSamples(k, 3) = sampleHeight-hTileSamples(k, 1)+1;
  hTileSamples(k, 4) = tileWidth;
  if(hTileSamples(k, 3) > 2*tileHeight)
    hTileSamples(k, 3) = 2*tileHeight;
  end
  
  hTileSamples(k, 3:4) = ...
      hTileSamples(k, 3:4) + hTileSamples(k, 1:2) - 1;
  
end

% initialize the results
resultTiles = ...
    zeros([numHColors*numHColors numVColors*numVColors sampleDepth]);

tileError = 0;

% generate individual wang tiles and pack them into the result
for sEdge = 1:numHColors
  sLoc = hTileSamples(sEdge, :);
  sLoc(3) = floor((sLoc(1)+sLoc(3))/2);
  sTile = sampleTexture(sLoc(1):sLoc(3), sLoc(2):sLoc(4), :);
      
  for eEdge = 1:numVColors
  eLoc = vTileSamples(eEdge, :);
  eLoc(4) = floor((eLoc(2)+eLoc(4))/2);
  eTile = sampleTexture(eLoc(1):eLoc(3), eLoc(2):eLoc(4), :);
  
    for nEdge = 1:numHColors
      nLoc = hTileSamples(nEdge, :);
      nLoc(1) = floor((nLoc(1)+nLoc(3))/2)+1;
      nTile = sampleTexture(nLoc(1):nLoc(3), nLoc(2):nLoc(4), :);
    
      for wEdge = 1:numVColors
        wLoc = vTileSamples(wEdge, :);
        wLoc(2) = floor((wLoc(2)+wLoc(4))/2)+1;
        wTile = sampleTexture(wLoc(1):wLoc(3), wLoc(2):wLoc(4), :);
     
        % get all source tiles, find the Wang Tile
        [oneTile, sePath, enPath, nwPath, wsPath, pathCost] = ...
            SingleWangTile(sTile, eTile, nTile, wTile, borderSize);
        
        indexLoc = ...
            [WangTileIndex(nEdge, sEdge) WangTileIndex(wEdge, eEdge)];
        
        indexLoc = (indexLoc-1) .* tileSize + 1;
        
        resultTiles(indexLoc(1):indexLoc(1)+tileSize(1)-1,...
                    indexLoc(2):indexLoc(2)+tileSize(2)-1, :) = ...
                    oneTile;
                
        tileError = tileError + pathCost;
      end
     
    end
    
  end
  
end

% done
return;
