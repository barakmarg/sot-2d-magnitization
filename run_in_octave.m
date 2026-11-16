function run_in_octave(script, lib_only)
% RUN_IN_OCTAVE  Run create_2D_Magnetization_map.m (or another copy of it) in GNU Octave.
%   MATLAB allows local functions at the END of a script; Octave needs them to be
%   defined before the script code calls them.  This launcher splits the script at its
%   "local functions" marker, defines the functions first, then runs the main part.
%
%   run_in_octave                      % runs create_2D_Magnetization_map.m
%   run_in_octave('my_copy.m')         % runs another copy (e.g. with a different MODE)
%   run_in_octave('', true)            % only define the functions (for interactive use)
%
%   In MATLAB, just run create_2D_Magnetization_map.m directly.
if nargin < 1 || isempty(script), script = 'create_2D_Magnetization_map.m'; end
if nargin < 2, lib_only = false; end
txt = fileread(script);
marker = sprintf('\n%%%% ======================= local functions =======================');
k = strfind(txt, marker);
assert(~isempty(k), 'local-functions marker not found in %s', script);
k = k(end);                                 % the active (uncommented) section
tmp = tempname();  mkdir(tmp);
lib  = fullfile(tmp, 'sot_lib_defs.m');
main = fullfile(tmp, 'sot_main_part.m');
fid = fopen(lib, 'w');   fprintf(fid, '1;\n%s\n', txt(k+1:end));  fclose(fid);
fid = fopen(main, 'w');  fprintf(fid, '%s\n', txt(1:k));          fclose(fid);
evalin('base', sprintf('source(''%s'');', lib));
if ~lib_only
    evalin('base', sprintf('source(''%s'');', main));
end
end
