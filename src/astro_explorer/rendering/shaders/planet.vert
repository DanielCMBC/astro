#version 330 core
// Shared planet vertex stage, used by rocky.frag and gas_giant.frag.
// Instanced: position(3) radius(1) color(3) emissive(1).

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 3) in vec3 instance_position;
layout(location = 4) in float instance_radius;
layout(location = 5) in vec3 instance_color;
layout(location = 6) in float instance_emissive;

uniform mat4 u_view_projection;
uniform vec3 u_camera_position;
uniform vec3 u_star_position;

out vec3 v_normal;
out vec2 v_uv;
out vec3 v_light_direction;
out vec3 v_view_direction;
out vec3 v_color;
out vec3 v_object_position;
out float v_emissive;

void main() {
    vec3 world_position = instance_position + in_position * instance_radius;

    v_normal = normalize(in_normal);
    v_uv = in_uv;
    v_object_position = in_position;
    v_light_direction = normalize(u_star_position - world_position);
    v_view_direction = normalize(u_camera_position - world_position);
    v_color = instance_color;
    v_emissive = instance_emissive;

    gl_Position = u_view_projection * vec4(world_position, 1.0);
}
