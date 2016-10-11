#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in float value;

uniform mat4 P;
uniform mat4 MV;

uniform float minValue;
uniform float maxValue;

varying vec4 color;

vec3 ColorMapHot(float cVal) {
  return vec3(
      cVal<0.20 ? 1.*cVal*5 : 1.,
      cVal<0.40 ? 0 : cVal < 0.80 ? (cVal-.4)*2.5 : 1.,
      cVal<0.80 ? 0 : (cVal-0.8)*5 );
}

void main() {
  gl_Position = P * MV * (vec4(pos,1.));
  if (value==value) {
    float c = (value-minValue)/(maxValue-minValue);
    color = vec4(ColorMapHot(c),1.);
  } else {
    color = vec4(0.,0.,0.,0.);
  }
}




