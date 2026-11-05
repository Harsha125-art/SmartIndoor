function Main()
    % SmartIndoor MATLAB Launcher
    % Automatically locates the Python interpreter and executes the SmartIndoor pipeline.
    projectDir = fileparts(mfilename('fullpath'));
    pythonExecutable = getenv('PYTHON_EXECUTABLE');
    
    if isempty(pythonExecutable)
        % Check if a local virtual environment exists in the project root
        venvPython = fullfile(projectDir, '.venv', 'Scripts', 'python.exe');
        if exist(venvPython, 'file') == 2
            pythonExecutable = venvPython;
        else
            pythonExecutable = 'python';
        end
    end

    scriptPath = fullfile(projectDir, 'main.py');
    % Quote executable and script paths to handle paths with spaces
    command = sprintf('"%s" "%s"', pythonExecutable, scriptPath);
    
    fprintf('=== SmartIndoor: Launching Python Experiment from MATLAB ===\n');
    fprintf('Project Directory : %s\n', projectDir);
    fprintf('Python Executable : %s\n\n', pythonExecutable);
    
    previousDir = pwd;
    cleanup = onCleanup(@() cd(previousDir));
    cd(projectDir);
    
    % Use '-echo' to stream stdout and stderr live to the MATLAB command window
    status = system(command, '-echo');
    
    if status ~= 0
        error('SmartIndoor experiment failed with exit code %d.', status);
    else
        fprintf('\n=== SmartIndoor experiment completed successfully! ===\n');
    end
end
