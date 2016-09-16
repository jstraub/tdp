#version 330

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 rgb;

uniform mat4 P;
uniform mat4 MV;

out vec4 FragColor;

// built in variables overview: 
// https://www.opengl.org/wiki/Built-in_Variable_(GLSL)

void main() {
  vec4 posOut = P*MV*vec4(pos,1);

  // fisheye effect:
  vec4 pos3D = MV*vec4(pos,1);
  posOut.xy = pos3D.xy / length(pos3D.xyz);

  posOut.y *= -1.;
  gl_Position = posOut; //vec4(pos3D,-1);

  //FragColor = vec4(1,0,0,1);
  FragColor = vec4(rgb,1.);
}


