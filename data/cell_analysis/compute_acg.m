function acg = compute_acg(times, binSize, duration)
% COMPUTE_ACG  Autocorrelogram of one spike train, in Hz.
%
%   acg = compute_acg(times, binSize, duration)
%
%   times     spike times in seconds (vector)
%   binSize   bin size in seconds (e.g. 0.001 for wide, 0.0005 for narrow)
%   duration  total width of the ACG in seconds (e.g. 1 for wide, 0.1 for narrow)
%
%   Returns a (2*halfBins+1) x 1 vector, halfBins = round(duration/binSize/2), centred on lag 0.
%   Each spike pair (both directions) is assigned to bin round(dt/binSize); counts are divided by
%   (n spikes * binSize), i.e. the same 'rate' normalisation as CellExplorer's CCG(...,'norm','rate').
%
%   This replaces CellExplorer's CCG/CCGHeart mex. The mex first rounds spike times to samples
%   (via 'Fs'), which shifts pairs near bin edges; working directly in seconds reproduces the ACGs
%   stored in spikes_struct.mat exactly, and needs no compiled code.

times = sort(times(:));
n = numel(times);
halfBins = round(duration/binSize/2);
counts = zeros(2*halfBins+1, 1);
if n < 2
    acg = counts;
    return
end

maxLag = (halfBins + 0.5) * binSize;
for k = 1:n-1
    d = times(1+k:end) - times(1:end-k);      % all pairs k spikes apart
    d = d(d <= maxLag);
    if isempty(d), break, end                 % sorted times: larger k only gives larger lags
    b = round(d / binSize);
    b = b(b <= halfBins);
    counts = counts + accumarray(halfBins + 1 + b, 1, size(counts)) ...
                    + accumarray(halfBins + 1 - b, 1, size(counts));
end
acg = counts / (n * binSize);
end
