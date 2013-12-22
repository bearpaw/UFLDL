function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------   

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% STEP 1: Computer Cost Function
rho = sparsityParam;
m = size(data, 2);          % number of training samples

z2 = W1 * data + repmat(b1, 1, m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2, 1, m);
h  = z3;
features = h;

rhopred = (1/m)*sum(a2, 2);

% compute cost
datacost = sum(0.5*(sum((h-data).^2)))/m;
penalty_W = (lambda/2) * sum([W1(:) ; W2(:)].^2);
penalty_sparsity = beta*sum(rho* log(rho./rhopred) + (1 - rho)* log((1-rho)./(1-rhopred)));


% overall cost
cost = datacost + penalty_W + penalty_sparsity;

%% STEP 2: Computer Gradient 
sparsity_delta = - rho ./ rhopred + (1 - rho) ./ (1 - rhopred);

delta3 = -(data - h);
delta2 = (W2'*delta3 + beta*repmat(sparsity_delta, 1, m)).* a2.*(1-a2);
    
W1grad = W1grad + (delta2*data');
W2grad = W2grad + (delta3*a2');
b1grad = b1grad + sum(delta2, 2);
b2grad = b2grad + sum(delta3, 2);



% lost

b1grad=b1grad/m;
b2grad=b2grad/m;
W1grad=W1grad/m+lambda*W1;
W2grad=W2grad/m+lambda*W2;



%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end