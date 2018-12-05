function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

% ======learning curves=======
for i = 1:m
    % training error
    X_get = X(1:i,:);
    y_get = y(1:i,:);
    [theta] = trainLinearReg(X_get, y_get, lambda);
    % make lambda=0 in the cost function
    [J_train, grad] = linearRegCostFunction(X_get, y_get, theta, 0);
    error_train(i) = J_train;
    
    % cross validation error
    [J_val, grad] = linearRegCostFunction(Xval, yval, theta, 0);
    error_val(i) = J_val;
end


% =======Last Optional: learning curves========
% error_train = zeros(m,50);
% error_val = zeros(m,50);
% for i = 1:m
%     % multiples times(50)
%     for j = 1:50
%         % ==========training set===========
%         % randomly select the examples
%         rand_num_train = randperm( size(X,1) );
%         X_rand_train = X(rand_num_train,:);
%         y_rand_train = y(rand_num_train,:);
%         
%         % training error
%         X_get = X_rand_train(1:i,:);
%         y_get = y_rand_train(1:i,:);
%         [theta] = trainLinearReg(X_get, y_get, lambda);
%         
%         % make lambda=0 in the cost function
%         [J_train, grad] = linearRegCostFunction(X_get, y_get, theta, 0);
%         error_train(i,j) = J_train;
%         
%         %==========cross validation set=========
%         %randomly select the examples
%         rand_num_val = randperm( size(Xval,1) );
%         X_rand_val = Xval(rand_num_val,:);
%         y_rand_val = yval(rand_num_val,:);
%         
%         % cross validation error
%         X_get_val = X_rand_val(1:i,:);
%         y_get_val = y_rand_val(1:i,:);
%         
%         % make lambda=0 in the cost function
%         [J_val, grad] = linearRegCostFunction(X_get_val, y_get_val, theta, 0);
%         error_val(i,j) = J_val;
%     end
% end
% 
% % get the average of the errores
% error_train = sum(error_train,2) / 50;
% error_val = sum(error_val,2) / 50;


% -------------------------------------------------------------

% =========================================================================

end
