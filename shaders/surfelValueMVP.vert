#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in float value;
layout (location = 2) in vec3 n;
layout (location = 3) in float r;

uniform mat4 MVP;
uniform float minValue;
uniform float maxValue;

out vec4 vPosC;
out vec4 vNC;
out float vRad;
out vec3 vColor;
out mat4 vMVP;

vec3 ColorMapHot(float cVal) {
  return vec3(
      cVal<0.20 ? 1.*cVal*5 : 1.,
      cVal<0.40 ? 0 : cVal < 0.80 ? (cVal-.4)*2.5 : 1.,
      cVal<0.80 ? 0 : (cVal-0.8)*5 );
}

void main() {
  // Transform into camera coordinates
  if ( pos.z > 10) {
    gl_Position = vec4(1000.f,1000.f,1000.f,1000.f);
  } else {
    gl_Position = MVP * vec4(pos, 1.f);
    if (value==value) {
      float c = (value-minValue)/(maxValue-minValue);
      vColor = ColorMapHot(c);
    } else {
      vColor = vec3(0.,0.,0.);
    }
    vMVP = MVP;
    vPosC = vec4(pos,1.f);
    vNC = vec4(n,1.f);
    vRad = r;
  }
}
