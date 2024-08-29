function x = prog(a, b, c)
    % solve_quadratic Solves the quadratic equation ax^2 + bx + c = 0
    % Returns the roots of the equation
    
    % Calculate the discriminant
    a = double(a);
    b = double(b);
    c = double(c);
    D = double(b^2 - 4*a*c);
    
    x(1) = double(-b + sqrt(D)) / (2*a);
    x(2) = double(-b - sqrt(D)) / (2*a);
end