#version 330 core
// Scientific region fragment stage.
//
// Deliberately flat: the band is a uniform wash of the colour it was given,
// with no gradient towards either edge. A gradient would read as confidence
// falling off across the zone, which is a scientific statement this model
// does not make - the conservative habitable zone has two hard boundaries,
// not a soft centre.

in vec4 v_color;

out vec4 frag_color;

void main() {
    frag_color = v_color;
}
