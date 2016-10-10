#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 value;

uniform mat4 P;
uniform mat4 MV;

// length
uniform float minValue;
uniform float maxValue;

varying vec4 color;

void main() {
  gl_Position = P * MV * (vec4(pos,1.));
  if (value.x==value.x) {
    vec3 c = vec3((value.x+maxValue)/(2.*maxValue),
        (value.y+maxValue)/(2.*maxValue),
        (value.z+maxValue)/(2.*maxValue));
    color = vec4(c,1.);
    //color = vec4(1.,0.,0.,1.);
  } else {
    color = vec4(0.,0.,0.,0.);
  }
}



