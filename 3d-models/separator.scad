// Box parameters (should match your existing box parameters)
length = 76;             // Outer length of the box
width = 37;              // Outer width of the box
wall_thickness = 2;      // Thickness of the walls
corner_radius = 5;       // Radius of the outer corners

// Separator parameters
separator_thickness = 2;           // Thickness of the separator
separator_clearance = 0.2;         // Clearance for proper fit

// Pillar parameters
pillar_radius = 2;               // Radius of the pillars
pillar_height = 12;                // Height of the pillars (as per your request)
pillar_margin = 1;                 // Margin from the edges

// Ventilation parameters
vent_line_width = 1;               // Width of the ventilation lines (made thinner)
vent_line_spacing = 3;             // Spacing between lines (center to center)

// Inner dimensions of the box
inner_length = length - 2 * wall_thickness;
inner_width  = width - 2 * wall_thickness;

// Inner corner radius
inner_corner_radius = corner_radius - wall_thickness;

// Separator dimensions
separator_length = inner_length - 2 * separator_clearance;
separator_width  = inner_width - 2 * separator_clearance;
separator_radius = inner_corner_radius - separator_clearance;

// Hole dimensions
hole_length = 56;  // Length of the hole
hole_width = 8;    // Width of the hole

// Create the separator with ventilation lines, pillars, and hole
module separator_with_ventilation_lines_and_pillars() {
    difference() {
        // Separator plate
        translate([0, 0, 0])
            linear_extrude(separator_thickness)
                rounded_rectangle(separator_length, separator_width, separator_radius);

        // Ventilation lines (code for this can be added as needed)

        // Subtract the hole
        translate([0, separator_width/2 - hole_width/2, 0])  // Centered along length, aligned to the edge on width
            linear_extrude(separator_thickness)
                square([hole_length, hole_width + 1], center = true);
    }

    // Pillars
    // Positions for the pillars (stepped back by pillar_margin from the edges)
    positions = [
        [ separator_length / 2 - pillar_margin - pillar_radius,  separator_width / 2 - pillar_margin - pillar_radius],
        [-separator_length / 2 + pillar_margin + pillar_radius,  separator_width / 2 - pillar_margin - pillar_radius],
        [-separator_length / 2 + pillar_margin + pillar_radius, -separator_width / 2 + pillar_margin + pillar_radius],
        [ separator_length / 2 - pillar_margin - pillar_radius, -separator_width / 2 + pillar_margin + pillar_radius]
    ];

    // Create pillars at each corner, connected to the separator
    for (pos = positions) {
        translate([pos[0], pos[1], -pillar_height])
            pillar(pillar_radius, pillar_height + separator_thickness);
    }
}

// Module to create a rounded rectangle
module rounded_rectangle(len, wid, rad) {
    rad = min(rad, len / 2, wid / 2); // Ensure radius does not exceed half the length or width
    offset(r = rad)
        square([len - 2 * rad, wid - 2 * rad], center = true);
}

// Module to create a pillar (cylinder)
module pillar(radius, height) {
    cylinder(h = height, r = radius, center = false);
}

// Render the separator with ventilation lines, pillars, and hole
separator_with_ventilation_lines_and_pillars();
