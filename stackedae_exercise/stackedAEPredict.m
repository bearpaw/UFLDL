function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
z1 = bsxfun(@plus, stack{1}.w * data, stack{1}.b);
% z1 = stack{1}.w * data + repmat(stack{1}.b, 1, size(data, 2));
a1 = sigmoid(z1);
z2 = bsxfun(@plus, stack{2}.w * a1, stack{2}.b);
% z2 = stack{2}.w * a1 + repmat(stack{2}.b, 1, size(data, 2));
a2 = sigmoid(z2);


M = softmaxTheta * a2;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
[v pred] = max(bsxfun(@rdivide, M, sum(M))); 







% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
