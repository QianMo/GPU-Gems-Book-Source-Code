%
% overlay an image over a Wang Tile
%
% function [result, cost] = OverlayWangTile(original, overlay, overlay_region)
% 
% Input: 
%
% original: the original image
% overlay: the overlay image. must have the same size as original.
% overlay_region: 4 vector specifying the region for overlay path cut.
%                 [row_min row_max col_min col_max]: all from image border
%
% Return:
% 
% result: the overlay result 
% cost: total path cost
%
% Li-Yi Wei
% 10/25/2004
%

function [result, cost] = OverlayWangTile(original, overlay, overlay_region)

% check arguments

if(nargin ~= 3)
  error('Wrong number of arguments');
end

if(ndims(original) ~= ndims(overlay))
  error('original and overlay must have the same dimensionality');
end

if(any(size(original) ~= size(overlay)))
  error('original and overlay must have the same size');
end

if(prod(size(overlay_region)) ~= 4)
  error('overlay_region must have 4 components');
end

if((overlay_region(2) < overlay_region(1)) | ...
      (overlay_region(4) < overlay_region(3)) )
  error('wrong overlay region specification');
end

[height width depth] = size(original);

row_min = overlay_region(1);
row_max = overlay_region(2);
col_min = overlay_region(3);
col_max = overlay_region(4);

% compute the 4 marker points for path start/end
n_row_min = row_min;
n_row_max = row_max;
s_row_min = height - row_max + 1;
s_row_max = height - row_min + 1;
w_col_min = col_min;
w_col_max = col_max;
e_col_min = width - col_max + 1;
e_col_max = width - col_min + 1;

marker_sw = [RandInt(s_row_min, s_row_max), RandInt(w_col_min, w_col_max)];
marker_se = [RandInt(s_row_min, s_row_max), RandInt(e_col_min, e_col_max)];
marker_ne = [RandInt(n_row_min, n_row_max), RandInt(e_col_min, e_col_max)];
marker_nw = [RandInt(n_row_min, n_row_max), RandInt(w_col_min, w_col_max)];

% compute cost surface
diff_image = abs(original - overlay);
costSurface = 0;

for k = 1:size(diff_image, 3)
  costSurface = costSurface + diff_image(:, :, k).^2;
end

if(0)
  costSurface(1:row_min-1, :) = inf;
  costSurface(end-row_min+2:end, :) = inf;
  costSurface(:, 1:col_min-1) = inf;
  costSurface(:, end-col_min+2:end) = inf;
  costSurface(row_max+1:end-row_max, col_max+1:end-col_max) = inf;
end

% compute paths and cost
[path_s, cost_s] = MinCostPath(costSurface, marker_sw, marker_se);
[path_e, cost_e] = MinCostPath(costSurface, marker_se, marker_ne);
[path_n, cost_n] = MinCostPath(costSurface, marker_ne, marker_nw);
[path_w, cost_w] = MinCostPath(costSurface, marker_nw, marker_sw);

% compute result and cost
cost = cost_s/size(path_s, 1) + cost_e/size(path_e, 1) + cost_n/size(path_n, 1) + cost_w/size(path_w, 1);

result = overlay;

result(marker_sw(1):end, 1:marker_sw(2), :) = ...
    original(marker_sw(1):end, 1:marker_sw(2), :);
result(marker_se(1):end, marker_se(2):end, :) = ...
    original(marker_se(1):end, marker_se(2):end, :);
result(1:marker_ne(1), marker_ne(2):end, :) = ...
    original(1:marker_ne(1), marker_ne(2):end, :);
result(1:marker_nw(1), 1:marker_nw(2), :) = ...
    original(1:marker_nw(1), 1:marker_nw(2), :);

for k = 1:size(path_s, 1)
  result(path_s(k, 1):end, path_s(k, 2), :) = ...
      original(path_s(k, 1):end, path_s(k, 2), :);
end
for k = 1:size(path_e, 1)
  result(path_e(k, 1), path_e(k, 2):end, :) = ...
      original(path_e(k, 1), path_e(k, 2):end, :);
end
for k = 1:size(path_n, 1)
  result(1:path_n(k, 1), path_n(k, 2), :) = ...
      original(1:path_n(k, 1), path_n(k, 2), :);
end
for k = 1:size(path_w, 1)
  result(path_w(k, 1), 1:path_w(k, 2), :) = ...
      original(path_w(k, 1), 1:path_w(k, 2), :);
end

% done
return;

% rand int utility
function [result] = RandInt(min, max)

result = round(rand([1])*(max-min)+min);
return;
