#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in float value;

uniform mat4 P;
uniform mat4 MV;

uniform float minValue;
uniform float maxValue;

varying vec4 color;

void main() {
  gl_Position = P * MV * (vec4(pos,1.));
  float c = (value-minValue)/(maxValue-minValue);
  color = vec4(c,c,c,1.);
}



