function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% select the values of the C and sigma
C_vect = [0.01 0.03 0.1 0.3 1 3 10 30];  
sigma_vect = [0.01 0.03 0.1 0.3 1 3 10 30];

% get the size of C and sigma
a = size(C_vect, 2);
b = size(sigma_vect, 2);

% vector for memoring the prediction error
pred_error = zeros(a, b);

% get the vector of memory
for i = 1:a
    for j = 1:b
        % get a C and sigma
        C = C_vect(i);
        sigma = sigma_vect(j);
        % get the model 
        model= svmTrain(X, y, C, ...
                @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        
        % get the predictions
        predictions = svmPredict(model, Xval);
        
        % get the prediction error
        error = mean(double(predictions ~=yval));
        
        %memory the error 
        pred_error(i,j) = error;
    end
end

% find the minimun prediction error
min_error = min(min(pred_error));
[i, j] = find(pred_error == min_error);

% get C and sigma
C = C_vect(i);
sigma = sigma_vect(j);

% =========================================================================

end
