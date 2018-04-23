%
% tiles.m
%
% a script for doing Wang Tiles stuff
%
% Li-Yi Wei
% 8/19/2003
%

clear all; close all;

%
% test WangTileSet
%

sampleSize = [200 200];
downSample = 1;
tileSize = [64 64];
overlayRegion = [8 24 8 24];
tileColors = [2 2 1];
numIterations = 10;
numOverlayIterations = numIterations;

blurryWeight = 1;

dump = 0;

borderSize = overlayRegion(1:2:end);

% put your input texture here
% here, I select 2 examples
% for purple_flowers, it is more random,
% to increase tile variety, I let numOverlayIterations = numIterations;
% for brown_bricks, it is more regular,
% so I avoid any random overlay by setting numOverlayIterations = 0;
sampleTexture = double(imread('purple_flowers.jpg'))/255; tileSize = [128 128]; numOverlayIterations = numIterations;
sampleTexture = double(imread('brown_bricks.jpg'))/255; tileSize = [128 128]; downSample = 2; numOverlayIterations = 0;

sampleSize = [size(sampleTexture, 1) size(sampleTexture, 2)] / downSample;

downStep = floor([size(sampleTexture, 1) size(sampleTexture, 2)] ./ sampleSize);

if(any(downStep > [1 1]))
  gfilt = namedFilter('binom5');
  gfilt = gfilt/sum(gfilt(:));
  for k = 1:size(sampleTexture, 3)
    downSampled(:, :, k) = corrDn(sampleTexture(:, :, k), gfilt, ...
        'repeat', downStep);
  end
                          
  sampleTexture = downSampled(1:sampleSize(1), 1:sampleSize(2), :);
end

if(0)
  % show random ramp image
  sampleTexture = rand([sampleSize 3]);
  
  for k = 1:size(sampleTexture, 1)
    sampleTexture(k, :, 1) = sampleTexture(k, :, 1) * k/size(sampleTexture, 1);
  end

  for k = 1:size(sampleTexture, 2)
    sampleTexture(:, k, 2) = sampleTexture(:, k, 2) * k/size(sampleTexture, 2);
  end

  sampleTexture(:, :, 3) = sampleTexture(:, :, 3)/sqrt(prod(sampleSize));
end

bestError = inf;
bestTiles = [];

for k = 1:numIterations
  [resultTiles, tilingError] = ...
      WangTileSet(sampleTexture, tileColors, tileSize, borderSize);

  if(tilingError < bestError)
    bestError = tilingError;
    bestTiles = resultTiles;
  end  
end

resultTiles = bestTiles;

% overlay
overlayCornerRegion = sampleSize - tileSize;

bestError = inf;
bestTiles = resultTiles;

for k = 1:numOverlayIterations
  numRows = size(resultTiles, 1)/tileSize(1);
  numCols = size(resultTiles, 2)/tileSize(2);
  
  overlayError = 0;
  
  for row = 1:numRows
    for col = 1:numCols
      tile = resultTiles((row-1)*tileSize(1)+1:row*tileSize(1),...
                         (col-1)*tileSize(2)+1:col*tileSize(2), :);
      
      overlayCorner = ceil(rand([1 2]) .* overlayCornerRegion);
      overlay = sampleTexture(...
          overlayCorner(1):overlayCorner(1)+tileSize(1)-1, ...
          overlayCorner(2):overlayCorner(2)+tileSize(2)-1, :);
      
      [result, tilingError] = OverlayWangTile(tile, overlay, overlayRegion);
      overlayError = overlayError + tilingError;
      
      resultTiles((row-1)*tileSize(1)+1:row*tileSize(1),...
                  (col-1)*tileSize(2)+1:col*tileSize(2), :) = result;
    end
  end
  
  if(overlayError < bestError)
    bestError = overlayError;
    bestTiles = resultTiles;
  end
end

resultTiles = bestTiles;

if(1)
  % flip the vertical tile locations for GL display
  newTiles = resultTiles;
  
  numRows = size(resultTiles, 1)/tileSize(1);
  numCols = size(resultTiles, 2)/tileSize(2);

  for row = 1:numRows
    for col = 1:numCols
      tile = resultTiles((row-1)*tileSize(1)+1:row*tileSize(1),...
          (col-1)*tileSize(2)+1:col*tileSize(2), :);
      
      new_row = mod(row, numRows)+1;
      newTiles((new_row-1)*tileSize(1)+1:new_row*tileSize(1),...
          (col-1)*tileSize(2)+1:col*tileSize(2), :) = tile;
    end
  end
  
  resultTiles = newTiles;
end

figure;
imshow(resultTiles);

if(dump)
  imwrite(resultTiles, 'tile_pack.ppm', 'Encoding', 'ASCII');
end

if(1)
  numRows = size(resultTiles, 1)/tileSize(1);
  numCols = size(resultTiles, 2)/tileSize(2);
  
  figure;
  
  for row = 1:numRows
    for col = 1:numCols
      tile = resultTiles((row-1)*tileSize(1)+1:row*tileSize(1),...
                         (col-1)*tileSize(2)+1:col*tileSize(2), :);
      
      subplot(numRows, numCols, (row-1)*numCols + col);
      imshow(tile);
      if(dump & 0)
        imwrite(tile, ['tile_' num2str(row) '_' num2str(col) '.ppm'], 'Encoding', 'ASCII');
      end
    end
  end
end

return;
