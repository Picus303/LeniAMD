#version 450

layout(location = 0) in vec2 positions;
layout(location = 1) in vec3 colors;

layout(location = 0) out vec3 outColors;

void main () {
	gl_Position = vec4(positions, 0.0, 1.0);
	outColors = colors;
	gl_PointSize = 1; // Pour les points
}