#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in float value;

uniform sampler2D labels;
uniform mat4 P;
uniform mat4 MV;
uniform float offset;

out vec2 uv;

void main() {
  gl_Position = P * MV * (vec4(pos,1.));
  if (value==value) {
    uv = vec2((mod(value+offset,256.)+0.5)/256.,((value+offset)/256.+1.)/256.);
  } else {
    uv = vec2(0.,0.);
  }
}




