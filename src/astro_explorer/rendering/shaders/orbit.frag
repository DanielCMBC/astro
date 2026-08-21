#version 330 core
// Orbit path fragment stage.
//
// A dashed path means at least one orbital element used to draw it was an
// assumption rather than a measurement. The renderer is told only the dash
// period; the reason belongs to the UI, which shows it in words.
//
// A dash period of zero means "solid", which is how a fully measured orbit
// is signalled without needing a separate draw call.

in float v_arclength;
in vec4 v_color;
in float v_dash_period;

out vec4 frag_color;

void main() {
    if (v_dash_period > 0.0) {
        float phase = fract(v_arclength / v_dash_period);
        if (phase > 0.55) {
            discard;
        }
    }
    frag_color = v_color;
}
