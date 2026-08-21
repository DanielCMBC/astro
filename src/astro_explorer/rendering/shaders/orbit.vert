#version 330 core
// Orbit paths, batched.
//
// Every orbit in the system lives in one vertex buffer and is drawn by a
// single indexed LINES call, so a six-planet system costs one draw call
// rather than six (roadmap section 25). That means the per-orbit style has
// to travel per vertex rather than as a uniform.

layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_arclength;
layout(location = 2) in vec4 in_color;
layout(location = 3) in float in_dash_period;

uniform mat4 u_view_projection;

out float v_arclength;
out vec4 v_color;
out float v_dash_period;

void main() {
    v_arclength = in_arclength;
    v_color = in_color;
    v_dash_period = in_dash_period;
    gl_Position = u_view_projection * vec4(in_position, 1.0);
}
