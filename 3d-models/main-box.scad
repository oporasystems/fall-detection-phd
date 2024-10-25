// Parameters
length = 76;            // Outer length of the box
width = 37;             // Outer width of the box
height = 25;            // Height of the box
wall_thickness = 2;     // Thickness of the walls (returned to 2mm)
corner_radius = 5;      // Radius of the outer corners

// USB Type-C hole parameters
usb_width = 10;         // Width of the USB Type-C hole
usb_height = 4;         // Height of the USB Type-C hole
usb_corner_radius = 1;  // Corner radius for the USB hole (optional)
usb_position_y = 0;     // Y position (centered vertically on the side wall)
usb_position_z = wall_thickness + usb_height / 2 + 2; // Near bottom edge

// Push button hole parameters
button_size = 6;        // Size of the push button holes (6mm x 6mm)
button1_position_x = -length / 4;  // Position of the first button hole
button2_position_x = length / 4;   // Position of the second button hole
button_position_z = (height / 2) - 4;    // Centered vertically

// New hole parameters
new_hole_width = 13;        // Width of the new hole (along the x-axis)
new_hole_height = 9;      // Height of the new hole (along the z-axis)
new_hole_position_x = 0;   // Position of the new hole along the x-axis (centered between existing holes)
new_hole_position_z = button_position_z - 1.5; // Vertical position (same as existing holes)

// Tolerance for proper subtraction
epsilon = 0.1;          // Small offset to ensure clean subtraction

// Calculate inner_corner_radius and handle corner_radius < wall_thickness
if (corner_radius < wall_thickness) {
    echo("Error: corner_radius must be greater than or equal to wall_thickness");
    corner_radius = wall_thickness; // Set corner_radius to wall_thickness to avoid negative values
}
inner_corner_radius = corner_radius - wall_thickness;

difference() {
    // Outer box with rounded corners
    linear_extrude(height)
        rounded_rectangle(length, width, corner_radius);

    // Inner box to create the hollow space
    translate([0, 0, wall_thickness])
        linear_extrude(height - wall_thickness)
            rounded_rectangle(length - 2 * wall_thickness, width - 2 * wall_thickness, inner_corner_radius);

    // USB Type-C hole on the longer side, near the bottom edge
    translate([length / 2 + epsilon, usb_position_y, usb_position_z])
        rotate([0, 0, 90]) // Rotate to align with the side wall
            usb_hole();

    // Push button holes on one of the sides
    // First button hole
    translate([button1_position_x, width / 2 + epsilon, button_position_z])
        cube([button_size, width + 2 * epsilon, button_size], center=true);

    // Second button hole
    translate([button2_position_x, width / 2 + epsilon, button_position_z])
        cube([button_size, width + 2 * epsilon, button_size], center=true);

    // New rectangular hole between the two existing holes
    translate([new_hole_position_x, width / 2 + epsilon, new_hole_position_z])
        cube([new_hole_width, width + 2 * epsilon, new_hole_height], center=true);
}

// Module to create a rounded rectangle
module rounded_rectangle(len, wid, rad) {
    rad = min(rad, len / 2, wid / 2); // Ensure radius does not exceed half the length or width
    offset(r=rad)
        square([len - 2 * rad, wid - 2 * rad], center=true);
}

// Module for the USB Type-C hole
module usb_hole() {
    if (usb_corner_radius > 0) {
        minkowski() {
            cube([usb_width - 2 * usb_corner_radius, usb_height - 2 * usb_corner_radius, wall_thickness + 2 * epsilon], center=true);
            sphere(r=usb_corner_radius);
        }
    } else {
        cube([usb_width, usb_height, wall_thickness + 2 * epsilon], center=true);
    }
}
