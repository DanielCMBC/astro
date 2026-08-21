#version 330 core
// Star fragment shader: limb darkening, granulation and HDR emission.
// The colour arrives already derived from effective temperature by the
// science layer; this shader never computes it.

in vec3 v_normal;
in vec3 v_view_direction;
in vec3 v_color;

uniform float u_limb_darkening;   // Eddington coefficient, typically 0.6
uniform float u_granulation;      // 0 disables the noise entirely
uniform float u_exposure;
uniform float u_time;

out vec4 frag_color;

// Cheap value noise; a real implementation would sample a 3D texture.
float hash(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash(i + vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

void main() {
    float mu = clamp(dot(normalize(v_normal), normalize(v_view_direction)), 0.0, 1.0);

    // Linear limb-darkening law: I(mu)/I(1) = 1 - a(1 - mu).
    float limb = 1.0 - u_limb_darkening * (1.0 - mu);

    float cells = 1.0;
    if (u_granulation > 0.0) {
        float n = noise(v_normal * 18.0 + vec3(0.0, 0.0, u_time * 0.05));
        cells = mix(1.0, 0.85 + 0.3 * n, u_granulation);
    }

    vec3 radiance = v_color * limb * cells * u_exposure;

    // Reinhard tone map: the star is an HDR emitter, the display is not.
    vec3 mapped = radiance / (radiance + vec3(1.0));
    frag_color = vec4(mapped, 1.0);
}
