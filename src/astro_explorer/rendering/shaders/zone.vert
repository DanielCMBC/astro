#version 330 core
// Scientific region overlays (the habitable zone), batched.
//
// A zone reaches the renderer as two closed loops that have already been
// sampled in display units. This stage does no geometry of its own: the
// band between the loops was triangulated on the CPU, so the shader has no
// opportunity to decide where a boundary lies.
//
// Colour travels per vertex for the same reason orbits do - every zone in
// the scene is drawn in one call, so a uniform could not vary between them.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;

uniform mat4 u_view_projection;

out vec4 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_view_projection * vec4(in_position, 1.0);
}
