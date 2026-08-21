#version 330 core
// Orbit path fragment stage.
//
// A dashed path means at least one orbital element used to draw it was an
// assumption rather than a measurement. The renderer is told only whether to
// dash; the reason belongs to the UI, which shows it in words.

in float v_arclength;

uniform vec4 u_color;
uniform bool u_dashed;
uniform float u_dash_period;

out vec4 frag_color;

void main() {
    if (u_dashed) {
        float phase = fract(v_arclength / max(u_dash_period, 1e-6));
        if (phase > 0.55) {
            discard;
        }
    }
    frag_color = u_color;
}
