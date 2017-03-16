#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 rgb;
layout (location = 2) in vec3 n;
layout (location = 3) in float r;

uniform mat4 MVP;

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
    vColor = rgb;
    vMVP = MVP;
    vPosC = vec4(pos,1.f);
    vNC = vec4(n,1.f);
    vRad = r;
  }
}
