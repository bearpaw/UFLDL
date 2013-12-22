function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% groundTruth is a k*n matrix, where each column is the groundtruth vector 
% of a sample (only one element equals to 1, and all the others are 0 in 
% each column)
M = theta * data;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
% h = M./repmat(sum(M), numClasses, 1); % h = bsxfun(@rdivide, M, sum(M)); 
h = bsxfun(@rdivide, M, sum(M)); 

datacost = -sum(sum((groundTruth.*log(h))))/numCases;
penalty = 0.5 * lambda * sum(theta(:).^2);

cost = datacost + penalty;

% Compute gradient
% Non vectorized version
% diff = (groundTruth - h);
% for j=1:numClasses
%     thetagrad(j, :) = -sum(data.* repmat(diff(j, :), size(data,1), 1), 2)/numCases;
% end
% thetagrad = thetagrad + lambda * theta;  % weight decay derivative

% vectorized version
thetagrad = -((groundTruth-h)*data')/numCases+lambda*theta;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

