function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 
EPSILON = 10^-4;
numgrad = zeros(size(theta));

for i = 1:length(theta(:))
    fprintf('%d | %d\n', i, length(theta(:)));
    tpos = theta;
    tneg = theta;
    tpos(i) = tpos(i)+EPSILON;
    tneg(i) = tneg(i)-EPSILON;
    
    numgrad(i) = 0.5 * ( J(tpos) - J(tneg)) / EPSILON;
end  

%% ---------------------------------------------------------------
end
