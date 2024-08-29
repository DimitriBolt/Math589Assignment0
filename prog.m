function x = prog(a, b, c)
    % solve_quadratic Solves the quadratic equation ax^2 + bx + c = 0
    % Returns the roots of the equation
    
    % Calculate the discriminant
    format long
    a = double(a);
    b = double(b);
    c = double(c);
    D = (b^2 - 4*a*c);
    
    x(1) = double(-b + sqrt(D)) / (2*a);
    x(1) = double(x(1));
    x(2) = double(-b - sqrt(D)) / (2*a);
    x(2) = double(x(2));
end