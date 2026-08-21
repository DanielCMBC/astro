#version 330 core
// Gas, hot and ultra-hot giants.
//
// Three features the roadmap asks for (section 15): procedural banding,
// turbulent cloud layers and limb haze. Hot giants additionally receive a
// temperature-dependent emission term and a strong day/night gradient; both
// are driven by uniforms the science layer computed, never by this shader.

in vec3 v_normal;
in vec2 v_uv;
in vec3 v_light_direction;
in vec3 v_view_direction;
in vec3 v_color;
in vec3 v_object_position;
in float v_emissive;

uniform float u_banding;          // 0 = featureless, 1 = strong bands
uniform float u_turbulence;
uniform float u_limb_haze;
uniform float u_day_night_contrast;
uniform float u_ambient;
uniform float u_time;

out vec4 frag_color;

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
    for (int octave = 0; octave < 6; ++octave) {
        total += amplitude * noise(p);
        p *= 2.07;
        amplitude *= 0.5;
    }
    return total;
}

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(v_light_direction);
    vec3 view = normalize(v_view_direction);

    // Latitude drives the banding; turbulence warps the band boundaries so
    // they are not perfect stripes.
    float latitude = normal.y;
    float warp = u_turbulence * (fbm(v_object_position * 3.0 + vec3(0.0, 0.0, u_time * 0.02)) - 0.5);
    float bands = sin((latitude + warp) * 18.0);
    bands = bands * 0.5 + 0.5;

    float clouds = fbm(v_object_position * 6.0 + vec3(u_time * 0.03, 0.0, 0.0));

    vec3 light_band = v_color * 1.25;
    vec3 dark_band = v_color * 0.7;
    vec3 albedo = mix(dark_band, light_band, mix(0.5, bands, u_banding));
    albedo = mix(albedo, albedo * (0.85 + 0.3 * clouds), u_turbulence);

    float n_dot_l = max(dot(normal, light), 0.0);
    // Hot giants have a sharp terminator; cool ones are softened by
    // scattering in the upper atmosphere.
    float day = mix(n_dot_l, pow(n_dot_l, 0.6), 1.0 - u_day_night_contrast);

    // Limb haze: forward scattering brightens the edge of the disc.
    float limb = 1.0 - max(dot(normal, view), 0.0);
    vec3 haze = v_color * pow(limb, 3.0) * u_limb_haze;

    vec3 color = albedo * day + albedo * u_ambient + haze;
    color += v_color * v_emissive * (0.35 + 0.65 * (1.0 - day));

    frag_color = vec4(color / (color + vec3(1.0)), 1.0);
}
