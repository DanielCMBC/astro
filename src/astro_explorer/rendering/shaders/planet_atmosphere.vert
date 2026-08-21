#version 330 core
// Vertex stage for the atmospheric shell. Identical to planet.vert except
// that the sphere is scaled outwards by the atmosphere's thickness.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 3) in vec3 instance_position;
layout(location = 4) in float instance_radius;

uniform mat4 u_view_projection;
uniform vec3 u_camera_position;
uniform vec3 u_star_position;
uniform float u_shell_scale;

out vec3 v_normal;
out vec3 v_light_direction;
out vec3 v_view_direction;

void main() {
    float radius = instance_radius * u_shell_scale;
    vec3 world_position = instance_position + in_position * radius;

    v_normal = normalize(in_normal);
    v_light_direction = normalize(u_star_position - world_position);
    v_view_direction = normalize(u_camera_position - world_position);

    gl_Position = u_view_projection * vec4(world_position, 1.0);
}
