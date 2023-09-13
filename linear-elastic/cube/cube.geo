fine = DefineNumber[0.06];

xmin = DefineNumber[0];
xmax = DefineNumber[1];
ymin = DefineNumber[0];
ymax = DefineNumber[1];
zmin = DefineNumber[0];
zmax = DefineNumber[1];

Point(1) = {xmin, ymin, zmin, fine};
Point(2) = {xmin, ymax, zmin, fine};
Point(3) = {xmax, ymax, zmin, fine};
Point(4) = {xmax, ymin, zmin, fine};
Point(5) = {xmin, ymin, zmax, fine};
Point(6) = {xmin, ymax, zmax, fine};
Point(7) = {xmax, ymax, zmax, fine};
Point(8) = {xmax, ymin, zmax, fine};


Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

Line Loop(13) = {1, 2, 3, 4};
Plane Surface(19) ={13}; // BOTTOM WALL
Line Loop(14) = {5, 6, 7, 8};
Plane Surface(20) ={14}; // TOP WALL
Line Loop(15) = {4, 9, -8, -12};
Plane Surface(21) ={15};
Line Loop(16) = {9, 5, -10, -1};
Plane Surface(22) ={16};
Line Loop(17) = {2, 11, -6, -10};
Plane Surface(23) ={17};
Line Loop(18) = {3, 12, -7, -11};
Plane Surface(24) ={18};

Surface Loop(25) = {19, 20, 21, 22, 23, 24};
Volume(26) ={25};

Physical Surface("1") = {19}; // BOTTOM
Physical Surface("2") = {20}; // TOP
Physical Surface("3") = {21};
Physical Surface("4") = {23};
Physical Surface("5") = {22};
Physical Surface("6") = {24};
Physical Volume("pde") = {26};