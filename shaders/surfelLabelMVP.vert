#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in float value;
layout (location = 2) in vec3 n;
layout (location = 3) in float r;

uniform mat4 MVP;
uniform sampler2D labels;
uniform float offset;

out vec4 vPosC;
out vec4 vNC;
out float vRad;
out vec3 vColor;
out mat4 vMVP;

void main() {
  // Transform into camera coordinates
  if ( pos.z > 10) {
    gl_Position = vec4(1000.f,1000.f,1000.f,1000.f);
  } else {
    gl_Position = MVP * vec4(pos, 1.f);
    vec2 uv;
    if (value==value) {
      uv = vec2((mod(value+offset,256.)+0.5)/256.,((value+offset)/256.+1.)/256.);
    } else {
      uv = vec2(0.,0.);
    }
    vColor = texture(labels,uv).xyz;
    vMVP = MVP;
    vPosC = vec4(pos,1.f);
    vNC = vec4(n,1.f);
    vRad = r;
  }
}
