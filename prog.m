function x = prog(a, b, c)
    % solve_quadratic Solves the quadratic equation ax^2 + bx + c = 0
    % Returns the roots of the equation
    
    % Calculate the discriminant
    D = b^2 - 4*a*c;
    
    x(1) = (-b + sqrt(D)) / (2*a);
    x(2) = (-b - sqrt(D)) / (2*a);
end