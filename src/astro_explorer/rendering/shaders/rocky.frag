#version 330 core
// Rocky and icy bodies. Lambert diffuse plus a GGX specular lobe, with an
// emissive term for bodies hot enough to glow on their own.
//
// The base colour arrives from the procedural material layer, which derives
// it from equilibrium temperature and bulk density. This shader adds no
// science of its own.

in vec3 v_normal;
in vec2 v_uv;
in vec3 v_light_direction;
in vec3 v_view_direction;
in vec3 v_color;
in vec3 v_object_position;
in float v_emissive;

uniform float u_roughness;
uniform float u_ambient;
uniform float u_terrain_strength;
uniform sampler2D u_albedo;
uniform bool u_use_texture;

out vec4 frag_color;

const float PI = 3.14159265359;

float hash(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
                   mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
                   mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

float fbm(vec3 p) {
    float total = 0.0;
    float amplitude = 0.5;
    for (int octave = 0; octave < 5; ++octave) {
        total += amplitude * noise(p);
        p *= 2.03;
        amplitude *= 0.5;
    }
    return total;
}

float ggx_distribution(float n_dot_h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(v_light_direction);
    vec3 view = normalize(v_view_direction);
    vec3 half_vector = normalize(light + view);

    vec3 albedo = v_color;
    if (u_use_texture) {
        albedo *= texture(u_albedo, v_uv).rgb;
    } else if (u_terrain_strength > 0.0) {
        // Procedural surface variation, clearly not a map of a real surface.
        float terrain = fbm(v_object_position * 4.0);
        albedo *= mix(1.0, 0.6 + 0.8 * terrain, u_terrain_strength);
    }

    float n_dot_l = max(dot(normal, light), 0.0);
    float n_dot_h = max(dot(normal, half_vector), 0.0);

    vec3 diffuse = albedo * n_dot_l / PI;
    float specular = ggx_distribution(n_dot_h, clamp(u_roughness, 0.05, 1.0)) * 0.04 * n_dot_l;

    vec3 color = diffuse + vec3(specular) + albedo * u_ambient;
    color += albedo * v_emissive;

    frag_color = vec4(color / (color + vec3(1.0)), 1.0);
}
