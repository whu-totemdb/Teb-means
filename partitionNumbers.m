function blocks = partitionNumbers(n, block_size)
    %PARTITIONNUMBERS Divides numbers from 1 to n into blocks of specified size.
    %
    %   blocks = partitionNumbers(n, block_size) partitions the integers
    %   from 1 to n into a cell array of blocks, each containing up to
    %   block_size elements. The last block may contain fewer elements
    %   if n is not perfectly divisible by block_size.
    %
    %   Inputs:
    %       n          - Total number of elements (positive integer)
    %       block_size - Desired size of each block (positive integer)
    %
    %   Output:
    %       blocks     - 1 x num_blocks cell array, each cell contains
    %                    a vector of indices for that block.
    
    % Validate inputs
    if nargin ~= 2
        error('partitionNumbers requires exactly two input arguments: n and block_size.');
    end
    if ~isscalar(n) || n <= 0 || n ~= floor(n)
        error('n must be a positive integer.');
    end
    if ~isscalar(block_size) || block_size <= 0 || block_size ~= floor(block_size)
        error('block_size must be a positive integer.');
    end
    
    % Calculate the number of full blocks and the size of the last block
    num_full_blocks = floor(n / block_size);
    remainder = mod(n, block_size);
    
    % Initialize the blocks cell array
    if remainder > 0
        num_blocks = num_full_blocks + 1;
    else
        num_blocks = num_full_blocks;
    end
    blocks = cell(1, num_blocks);
    
    % Assign indices to each block
    for k = 1:num_full_blocks
        start_idx = (k - 1) * block_size + 1;
        end_idx = k * block_size;
        blocks{k} = start_idx:end_idx;
    end
    
    % Assign remaining indices to the last block, if any
    if remainder > 0
        blocks{end} = (num_full_blocks * block_size + 1):n;
    end
end
