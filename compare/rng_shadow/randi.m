function varargout = randi(varargin)
% Test-only shadow of Octave's builtin randi: identical output, but every draw is also
% appended to the global log SOT_RNG_LOG.randi (column-major), so that the Python port can
% replay exactly the same random numbers.  Only on the path inside compare/.
global SOT_RNG_LOG
if ~all(cellfun(@isnumeric, varargin))        % e.g. rand('twister', s) from rng()
    [varargout{1:nargout}] = builtin('randi', varargin{:});
    return
end
[varargout{1:max(nargout,1)}] = builtin('randi', varargin{:});
if isstruct(SOT_RNG_LOG)
    SOT_RNG_LOG.randi{end+1} = double(varargout{1}(:));
end
end
