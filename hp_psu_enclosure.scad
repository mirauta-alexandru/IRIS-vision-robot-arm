// ============================================================
// Carcasă pentru Sursa HP Compaq 6200/8200 PRO SFF
// Dimensiuni sursă: 160 x 95 x 145 mm (W x H x D)
// ============================================================

// --- Parametri principali (modifică aici) ---
psu_w = 160;    // lățime sursă
psu_h = 95;     // înălțime sursă
psu_d = 145;    // adâncime sursă

wall = 2.5;         // grosime perete
tolerance = 1.0;    // joc pe fiecare latură
lip_h = 5;          // buza capacului

// Dimensiuni interne (sursă + toleranță)
inner_w = psu_w + 2 * tolerance;
inner_h = psu_h + 2 * tolerance;
inner_d = psu_d + 2 * tolerance;

// Dimensiuni externe
outer_w = inner_w + 2 * wall;
outer_h = inner_h + 2 * wall;
outer_d = inner_d + 2 * wall;

// Parametri montare
screw_d = 3.2;          // diametru gaură șurub M3
screw_head_d = 6;       // diametru cap șurub
mount_pillar = 8;       // lățime stâlp montare
mount_h = inner_h;      // înălțime stâlpi

// Parametri ventilație
vent_slot_w = 2;        // lățime slot ventilație
vent_slot_gap = 3;      // spațiu între sloturi
fan_d = 80;             // diametru ventilator

// Parametri slot cabluri (partea de sus-spate, unde ies firele)
cable_slot_w = 50;      // lățime slot cabluri
cable_slot_h = 30;      // înălțime slot cabluri
cable_r = 3;            // raza rotunjire slot cabluri

// Parametri conector IEC C14 (priză curent, pe fața frontală lângă ventilator)
// Decupaj standard IEC C14: ~28 x 20 mm
iec_w = 30;             // lățime decupaj IEC (cu toleranță)
iec_h = 22;             // înălțime decupaj IEC (cu toleranță)
iec_r = 2;              // raza rotunjire colțuri IEC
// Poziția: colțul din dreapta-jos al feței frontale (privind din față)
iec_offset_x = outer_w - wall - iec_w - 8;  // de la dreapta
iec_offset_z = wall + 5;                      // de la fund

// --- Raza colțuri ---
corner_r = 3;

// ============================================================
// MODULE
// ============================================================

// Cutie cu colțuri rotunjite
module rounded_box(w, h, d, r) {
    translate([r, r, 0])
    minkowski() {
        cube([w - 2*r, d - 2*r, h - r]);
        cylinder(r=r, h=r, $fn=30);
    }
}

// Grila ventilație circulară (pentru fan)
module fan_vent(diameter, slot_w, slot_gap) {
    n_slots = floor(diameter / (slot_w + slot_gap));
    for (i = [0 : n_slots - 1]) {
        y_pos = i * (slot_w + slot_gap) - (n_slots * (slot_w + slot_gap)) / 2;
        // Calculăm lățimea slotului la această poziție (cerc)
        half_chord = sqrt(max(0, (diameter/2)*(diameter/2) - y_pos*y_pos));
        if (half_chord > 2) {
            translate([-wall-1, y_pos - slot_w/2, 0])
            cube([wall + 2, slot_w, half_chord * 2]);
        }
    }
}

// Grila ventilație dreptunghiulară (pentru lateral)
module side_vent(width, height, slot_w, slot_gap) {
    n_slots = floor(height / (slot_w + slot_gap));
    for (i = [0 : n_slots - 1]) {
        z_pos = i * (slot_w + slot_gap);
        if (z_pos + slot_w < height) {
            translate([0, 0, z_pos])
            cube([wall + 2, width, slot_w]);
        }
    }
}

// Stâlp de montare cu gaură de șurub
module mount_pillar_mod(h) {
    difference() {
        cylinder(d=mount_pillar, h=h, $fn=30);
        translate([0, 0, -1])
        cylinder(d=screw_d, h=h+2, $fn=20);
    }
}

// Slot pentru cabluri cu colțuri rotunjite (tăiat pe axa Y)
module cable_slot(w, h, r) {
    hull() {
        translate([r, 0, r]) rotate([-90, 0, 0]) cylinder(r=r, h=wall+2, $fn=20);
        translate([w-r, 0, r]) rotate([-90, 0, 0]) cylinder(r=r, h=wall+2, $fn=20);
        translate([r, 0, h-r]) rotate([-90, 0, 0]) cylinder(r=r, h=wall+2, $fn=20);
        translate([w-r, 0, h-r]) rotate([-90, 0, 0]) cylinder(r=r, h=wall+2, $fn=20);
    }
}

// Decupaj conector IEC C14 cu colțuri rotunjite (tăiat pe axa Y - perete frontal)
module iec_cutout(w, h, r) {
    hull() {
        translate([r, -1, r]) rotate([-90, 0, 0]) cylinder(r=r, h=wall+2, $fn=20);
        translate([w-r, -1, r]) rotate([-90, 0, 0]) cylinder(r=r, h=wall+2, $fn=20);
        translate([r, -1, h-r]) rotate([-90, 0, 0]) cylinder(r=r, h=wall+2, $fn=20);
        translate([w-r, -1, h-r]) rotate([-90, 0, 0]) cylinder(r=r, h=wall+2, $fn=20);
    }
    // Găuri pentru șuruburile de fixare IEC (distanță 40mm între ele, centrate)
    translate([w/2 - 20, -1, h/2])
    rotate([-90, 0, 0]) cylinder(d=screw_d, h=wall+2, $fn=20);
    translate([w/2 + 20, -1, h/2])
    rotate([-90, 0, 0]) cylinder(d=screw_d, h=wall+2, $fn=20);
}

// ============================================================
// BAZA CARCASEI
// ============================================================
module base() {
    difference() {
        // Peretii exteriori
        rounded_box(outer_w, outer_h, outer_d, corner_r);

        // Golire interioară (fără capac sus)
        translate([wall, wall, wall])
        cube([inner_w, inner_d, inner_h + wall + 1]);

        // --- Slot cabluri (partea de sus-spate, unde ies firele din sursă) ---
        translate([
            (outer_w - cable_slot_w) / 2,
            outer_d - wall - 1,
            wall + inner_h - cable_slot_h - 5
        ])
        cable_slot(cable_slot_w, cable_slot_h, cable_r);

        // --- Decupaj conector IEC C14 (peretele frontal, dreapta-jos) ---
        translate([iec_offset_x, 0, iec_offset_z])
        iec_cutout(iec_w, iec_h, iec_r);

        // --- Ventilație laterală stânga ---
        translate([-1, wall + 15, wall + 10])
        side_vent(inner_d - 30, inner_h - 20, vent_slot_w, vent_slot_gap);

        // --- Ventilație laterală dreapta ---
        translate([outer_w - wall, wall + 15, wall + 10])
        side_vent(inner_d - 30, inner_h - 20, vent_slot_w, vent_slot_gap);

        // --- Ventilație frontală (zona ventilator, centrat dar decalat stânga
        //     pentru a lăsa loc conectorului IEC din dreapta) ---
        translate([outer_w/2 - 10, -1, wall + inner_h/2])
        rotate([-90, 0, 0])
        fan_vent(fan_d, vent_slot_w, vent_slot_gap);

        // --- Ventilație fund ---
        for (i = [0 : floor((inner_w - 20) / (vent_slot_w + vent_slot_gap))]) {
            translate([
                wall + 10 + i * (vent_slot_w + vent_slot_gap),
                wall + 10,
                -1
            ])
            cube([vent_slot_w, inner_d - 20, wall + 2]);
        }
    }

    // --- Stâlpi de montare în colțuri ---
    // Colț stânga-față
    translate([wall + mount_pillar/2, wall + mount_pillar/2, wall])
    mount_pillar_mod(mount_h);

    // Colț dreapta-față
    translate([outer_w - wall - mount_pillar/2, wall + mount_pillar/2, wall])
    mount_pillar_mod(mount_h);

    // Colț stânga-spate
    translate([wall + mount_pillar/2, outer_d - wall - mount_pillar/2, wall])
    mount_pillar_mod(mount_h);

    // Colț dreapta-spate
    translate([outer_w - wall - mount_pillar/2, outer_d - wall - mount_pillar/2, wall])
    mount_pillar_mod(mount_h);

    // --- Praguri de ghidaj pentru capac ---
    // Prag stânga
    translate([wall - 0.5, wall, inner_h + wall - lip_h])
    cube([1.5, inner_d, lip_h]);

    // Prag dreapta
    translate([outer_w - wall - 1, wall, inner_h + wall - lip_h])
    cube([1.5, inner_d, lip_h]);

    // Prag față
    translate([wall, wall - 0.5, inner_h + wall - lip_h])
    cube([inner_w, 1.5, lip_h]);

    // Prag spate
    translate([wall, outer_d - wall - 1, inner_h + wall - lip_h])
    cube([inner_w, 1.5, lip_h]);
}

// ============================================================
// CAPAC
// ============================================================
module lid() {
    lid_tolerance = 0.3;
    lid_inner_w = inner_w - 2 * lid_tolerance;
    lid_inner_d = inner_d - 2 * lid_tolerance;

    difference() {
        union() {
            // Placa capacului
            rounded_box(outer_w, wall, outer_d, corner_r);

            // Buza interioară (se introduce în bază)
            translate([wall + lid_tolerance, wall + lid_tolerance, -lip_h + wall])
            cube([lid_inner_w, lid_inner_d, lip_h]);
        }

        // Ventilație pe capac
        for (i = [0 : floor((inner_w - 20) / (vent_slot_w + vent_slot_gap))]) {
            translate([
                wall + 10 + i * (vent_slot_w + vent_slot_gap),
                wall + 10,
                -1
            ])
            cube([vent_slot_w, inner_d - 20, wall + 2]);
        }

        // Găuri șurub pe capac (aliniate cu stâlpii)
        translate([wall + mount_pillar/2, wall + mount_pillar/2, -1])
        cylinder(d=screw_d, h=wall+2, $fn=20);

        translate([outer_w - wall - mount_pillar/2, wall + mount_pillar/2, -1])
        cylinder(d=screw_d, h=wall+2, $fn=20);

        translate([wall + mount_pillar/2, outer_d - wall - mount_pillar/2, -1])
        cylinder(d=screw_d, h=wall+2, $fn=20);

        translate([outer_w - wall - mount_pillar/2, outer_d - wall - mount_pillar/2, -1])
        cylinder(d=screw_d, h=wall+2, $fn=20);

        // Îngropare capete șuruburi
        translate([wall + mount_pillar/2, wall + mount_pillar/2, wall - 1.5])
        cylinder(d=screw_head_d, h=2, $fn=20);

        translate([outer_w - wall - mount_pillar/2, wall + mount_pillar/2, wall - 1.5])
        cylinder(d=screw_head_d, h=2, $fn=20);

        translate([wall + mount_pillar/2, outer_d - wall - mount_pillar/2, wall - 1.5])
        cylinder(d=screw_head_d, h=2, $fn=20);

        translate([outer_w - wall - mount_pillar/2, outer_d - wall - mount_pillar/2, wall - 1.5])
        cylinder(d=screw_head_d, h=2, $fn=20);
    }
}

// ============================================================
// AFIȘARE
// ============================================================

// Baza
color("DimGray", 0.9)
base();

// Capacul (deplasat în sus pentru vizualizare)
color("SlateGray", 0.7)
translate([0, 0, outer_h + 15])
lid();

// --- Text informativ ---
echo(str("=== Carcasă HP PSU 6200/8200 PRO SFF ==="));
echo(str("Dimensiuni externe: ", outer_w, " x ", outer_h, " x ", outer_d, " mm"));
echo(str("Dimensiuni interne: ", inner_w, " x ", inner_h, " x ", inner_d, " mm"));
echo(str("Grosime perete: ", wall, " mm"));
echo(str("Toleranță: ", tolerance, " mm"));
