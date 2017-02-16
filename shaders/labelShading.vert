#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in float value;

uniform sampler2D labels;
uniform mat4 P;
uniform mat4 MV;

uniform float minValue;
uniform float maxValue;

out vec2 uv;

void main() {
  gl_Position = P * MV * (vec4(pos,1.));
  if (value==value) {
//    float c = 10000.*(value-minValue)/(maxValue-minValue);
//    uv = vec2((c%100)*0.01,(c/100)*0.01);
    float c = value-minValue;
    uv = vec2((mod(value,256.)+0.5)/256.,(value/256.+1.)/256.);
  } else {
    uv = vec2(0.,0.);
  }
}




