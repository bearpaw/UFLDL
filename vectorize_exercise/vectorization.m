% Examples of Vectorization
% source: http://ufldl.stanford.edu/wiki/index.php/Logistic_Regression_Vectorization_Example
% Author: platero.yang@gmail.com
function vectorization
n = 3;      % n-dim feature vector
m = 10;     % m samples

x = randi(10, n+1, m);
y = randi(2, 1, m)-1;
theta = randn([n+1, 1]);

% Implementation 1
grad1 = zeros(n+1,1);
for i=1:m,
  h = sigmoid(theta'*x(:,i));
  temp = y(i) - h; 
  for j=1:n+1,
    grad1(j) = grad1(j) + temp * x(j,i); 
  end;
end;



% Implementation 2 
grad2 = zeros(n+1,1);
for i=1:m,
  grad2 = grad2 + (y(i) - sigmoid(theta'*x(:,i)))* x(:,i);
end;

% Implementation 3
grad3 = x * (y- sigmoid(theta'*x))';

% Display result
grad1 
grad2
grad3
end


%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end