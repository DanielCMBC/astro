#version 330 core
// Star vertex shader. Instanced: one draw call for every star in view
// (roadmap section 25 - avoid per-object Python draw overhead).

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

// Per-instance: position(3) radius(1) color(3)
layout(location = 3) in vec3 instance_position;
layout(location = 4) in float instance_radius;
layout(location = 5) in vec3 instance_color;

uniform mat4 u_view_projection;
uniform vec3 u_camera_position;

out vec3 v_normal;
out vec3 v_view_direction;
out vec3 v_color;

void main() {
    vec3 world_position = instance_position + in_position * instance_radius;
    v_normal = normalize(in_normal);
    v_view_direction = normalize(u_camera_position - world_position);
    v_color = instance_color;
    gl_Position = u_view_projection * vec4(world_position, 1.0);
}
