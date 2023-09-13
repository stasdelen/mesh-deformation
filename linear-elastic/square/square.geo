fine = DefineNumber[0.03];

xmin = DefineNumber[0];
xmax = DefineNumber[1];
ymin = DefineNumber[0];
ymax = DefineNumber[1];

Point(1) = {xmin, ymin, 0, fine};
Point(2) = {xmin, ymax, 0, fine};
Point(3) = {xmax, ymax, 0, fine};
Point(4) = {xmax, ymin, 0, fine};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(9) = {1, 2, 3, 4};
Plane Surface(11) ={9};

Physical Curve("blr") = {1, 3, 4};
Physical Curve("top") = {2};
Physical Surface("pde") = {11};