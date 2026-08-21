#version 330 core
// Orbit paths, uploaded once per orbit as a single line strip.
// The arc-length parameter travels with each vertex so the fragment stage
// can dash a path whose elements were assumed rather than measured.

layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_arclength;

uniform mat4 u_view_projection;

out float v_arclength;

void main() {
    v_arclength = in_arclength;
    gl_Position = u_view_projection * vec4(in_position, 1.0);
}
