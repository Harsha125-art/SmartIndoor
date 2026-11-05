function Main()
    % SmartIndoor: runs the whole project without intervention and displays the results.
    % The first run creates a Python environment in .venv and installs the packages,
    % which needs Python 3.10 or newer and an internet connection.
    % Set the PYTHON_EXECUTABLE environment variable to use a specific Python instead.
    projectDir = fileparts(mfilename('fullpath'));
    previousDir = pwd;
    cleanup = onCleanup(@() cd(previousDir));
    cd(projectDir);

    fprintf('=== SmartIndoor ===\nProject directory: %s\n\n', projectDir);

    python = getenv('PYTHON_EXECUTABLE');
    if isempty(python)
        python = prepareEnvironment(projectDir);
    end
    fprintf('Python: %s\n\n', python);

    status = system(sprintf('"%s" main.py', python), '-echo');
    if status ~= 0
        error('SmartIndoor failed with exit code %d.', status);
    end

    displayResults(projectDir);
    fprintf('\n=== SmartIndoor completed successfully ===\n');
end


function python = prepareEnvironment(projectDir)
    if ispc
        python = fullfile(projectDir, '.venv', 'Scripts', 'python.exe');
    else
        python = fullfile(projectDir, '.venv', 'bin', 'python');
    end
    if exist(python, 'file') == 2
        return;
    end

    base = findPython();
    fprintf('Creating a Python environment with %s...\n', base);
    if system(sprintf('"%s" -m venv .venv', base), '-echo') ~= 0
        error('Could not create the Python environment.');
    end
    fprintf('Installing packages. This can take a few minutes...\n');
    if system(sprintf('"%s" -m pip install -r requirements.txt', python), '-echo') ~= 0
        error('Could not install the packages in requirements.txt.');
    end
end


function base = findPython()
    % MATLAB may not inherit the shell PATH, so common install locations are also tried.
    candidates = {'python3', 'python', '/opt/homebrew/bin/python3', '/usr/local/bin/python3'};
    check = '"%s" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"';
    for index = 1:numel(candidates)
        [status, ~] = system(sprintf(check, candidates{index}));
        if status == 0
            base = candidates{index};
            return;
        end
    end
    error('Python 3.10 or newer was not found. Install it or set PYTHON_EXECUTABLE.');
end


function displayResults(projectDir)
    finalDir = fullfile(projectDir, 'output', 'smartindoor', 'final');

    fprintf('\n=== Official test file: overall ===\n');
    disp(readtable(fullfile(finalDir, 'test_metrics.csv')));
    fprintf('\n=== Official test file: per class ===\n');
    disp(readtable(fullfile(finalDir, 'per_class_metrics.csv')));
    fprintf('\n=== Official test file: confusion matrix ===\n');
    disp(readtable(fullfile(finalDir, 'confusion_matrix.csv'), 'ReadRowNames', true, ...
        'VariableNamingRule', 'preserve'));
    fprintf('\n=== Official test file: coordinate error, meters ===\n');
    disp(readtable(fullfile(finalDir, 'coordinate_test_results.csv')));
    fprintf('\n=== 5-fold person-grouped CV: mean and SD per setup ===\n');
    disp(readtable(fullfile(projectDir, 'output', 'smartindoor', 'architecture_comparison.csv')));

    if usejava('desktop')
        figure('Name', 'SmartIndoor confusion matrix');
        imshow(imread(fullfile(finalDir, 'confusion_matrix.png')));
    end
end
