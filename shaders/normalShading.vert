#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 rgb;

uniform mat4 P;
uniform mat4 MV;

varying vec4 FragColor;

void main() {
  gl_Position = (vec4(pos,1.));
  FragColor = vec4(rgb,1.);
}


