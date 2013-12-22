function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% a1 = data;
z2 = bsxfun(@plus, stack{1}.w * data, stack{1}.b);
z2 = stack{1}.w * data + repmat(stack{1}.b, 1, size(data, 2));
a2 = sigmoid(z2);
z3 = bsxfun(@plus, stack{2}.w * a2, stack{2}.b);
a3 = sigmoid(z3);


M = softmaxTheta * a3;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
h = bsxfun(@rdivide, M, sum(M)); 


m = size(data,2);

datacost = -sum(sum((groundTruth.*log(h))))/m;
penalty = 0.5 * lambda * sum(softmaxTheta(:).^2);

cost = datacost + penalty;

% Gradient
softmaxThetaGrad = -((groundTruth-h)*a3')/m+lambda*softmaxTheta;

delta3 = -(softmaxTheta'*(groundTruth-h)).*a3.*(1-a3);
delta2 = (stack{2}.w'*delta3).* a2.*(1-a2);
    
stackgrad{1}.w = stackgrad{1}.w + (delta2*data');
stackgrad{2}.w = stackgrad{2}.w + (delta3*a2');
stackgrad{1}.b = stackgrad{1}.b + sum(delta2, 2);
stackgrad{2}.b = stackgrad{2}.b + sum(delta3, 2);

stackgrad{1}.b = stackgrad{1}.b/m;
stackgrad{2}.b = stackgrad{2}.b/m;
stackgrad{1}.w = stackgrad{1}.w/m;
stackgrad{2}.w = stackgrad{2}.w/m;









% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function sigmprime = sigmoidprime(x)
    sigm = sigmoid(x);
    sigmprime = sigm.*(1 - sigm);
end

function g = sigmoidGradient(z)
g=zeros(size(z));
g=sigmoid(z).*(1-sigmoid(z));
end
