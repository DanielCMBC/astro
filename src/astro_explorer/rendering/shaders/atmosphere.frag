#version 330 core
// Atmospheric shell, drawn as a slightly larger back-face-culled sphere.
//
// Roadmap section 15 is explicit that physically specific atmospheric
// behaviour must only be enabled when it is scientifically justified. The
// scale height, optical depth and scattering coefficients are all uniforms:
// when the science layer has no measured atmosphere it passes
// u_enabled = false and the shell renders as nothing at all, rather than
// inventing a blue haze around a planet with no known atmosphere.

in vec3 v_normal;
in vec3 v_light_direction;
in vec3 v_view_direction;

uniform bool u_enabled;
uniform vec3 u_rayleigh;       // wavelength-dependent scattering, per unit depth
uniform float u_mie;
uniform float u_mie_anisotropy; // Henyey-Greenstein g
uniform float u_optical_depth;
uniform float u_scale_height_fraction;

out vec4 frag_color;

float henyey_greenstein(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265359 * pow(max(denom, 1e-4), 1.5));
}

void main() {
    if (!u_enabled) {
        discard;
    }

    vec3 normal = normalize(v_normal);
    vec3 light = normalize(v_light_direction);
    vec3 view = normalize(v_view_direction);

    // Path length through the shell grows towards the limb.
    float mu = max(dot(normal, view), 0.0);
    float path = u_optical_depth / max(mu, 0.05);
    path *= u_scale_height_fraction;

    float cos_theta = dot(light, -view);

    // Rayleigh phase function, 3/16pi (1 + cos^2 t).
    float rayleigh_phase = 0.05968310365 * (1.0 + cos_theta * cos_theta);
    float mie_phase = henyey_greenstein(cos_theta, clamp(u_mie_anisotropy, -0.95, 0.95));

    float illumination = max(dot(normal, light), 0.0);
    vec3 scattered = (u_rayleigh * rayleigh_phase + vec3(u_mie * mie_phase)) * path * illumination;

    float transmittance = exp(-path);
    float alpha = clamp(1.0 - transmittance, 0.0, 1.0);

    frag_color = vec4(scattered / (scattered + vec3(1.0)), alpha);
}
