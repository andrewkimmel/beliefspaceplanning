clear all
syms theta1 theta2 theta1dot theta2dot u
syms m lc lc2 l2 I1 I2 l g

% phi = [t1, t2].';
% dphi = [dt1, dt2].';

M = sym('M', [2,2]);
M(1,1) = m * lc2 + m * (l2 + lc2 + 2 * l * lc * cos(theta2)) + I1 + I2;
M(2,2) = m * lc2 + I2;
M(1,2) = m * (lc2 + l * lc * cos(theta2)) + I2;
M(2,1) = M(1,2);

C = sym('C', [2,1]);
C(1) = -m * l * lc * theta2dot * theta2dot * sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * sin(theta2));
C(2) = m * l * lc * theta1dot * theta1dot * sin(theta2);
G = sym('G', [2,1]);
G(1) = (m * lc + m * l) * g * cos(theta1) + (m * lc * g * cos(theta1 + theta2));
G(2) = m * lc * g * cos(theta1 + theta2);

x = [theta1; theta2; theta1dot; theta2dot];
% a = [0;0;0;u];

% f = [dphi; -M\(C+G)];
% gu = [0; 0; M\[0;u]];

u1 = -1*0.1*theta1dot;
u2 = u - 1*0.1*theta2dot;
ddx = [theta1dot; theta2dot;
    (M(2,2) * (u1 - C(1) - G(1)) - M(1,2) * (u2 - C(2) - G(2))) / (M(1,1) * M(2,2) - M(1,2) * M(2,1));
       (M(1,1) * (u2 - C(2) - G(2)) - M(2,1) * (u1 - C(1) - G(1))) / (M(1,1) * M(2,2) - M(1,2) * M(2,1))];
A = jacobian(ddx, x);
B = jacobian(ddx, u);
% A = subs(A, {theta1, theta2, theta1dot, theta2dot}, {0,0,0,0});
% % A = subs(A, {l1,l2,m1,m2, g}, {1,1,1,1,10});
A = simplify(A);
B = simplify(B);

%%
% a = -M\(C+G) + M\[u1;u2];
a = G - [0;u];
% ua1 = solve(a(1), u);
% ua2 = solve(a(2), u);
