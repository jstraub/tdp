#version 330

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 rgb;

uniform vec4 cam; // fu fv uc vc
uniform float w;
uniform float h;

uniform float near;
uniform float far;

uniform mat4 P;
uniform mat4 MV;

out vec4 FragColor;

// built in variables overview: 
// https://www.opengl.org/wiki/Built-in_Variable_(GLSL)

void main() {
  vec4 posOut = P*MV*vec4(pos,1);
  vec4 pos3D = MV*vec4(pos,1);
  posOut.xy = pos3D.xy / length(pos3D.xyz);
  //vec3 posImg = vec3(
  //    pos3D.x/pos3D.z*cam.x, 
  //    pos3D.y/pos3D.z*cam.y,
  //    2*(pos3D.z-near)/(far-near)-1.);
  //vec3 posHom = vec3(
  //    2*posImg.x/w,
  //    2*posImg.y/h,
  //    posImg.z);

  //pos3D.z = 2*((pos3D.z-near)/(far-near))-1.;
  // homogeneous coordinates of output vertex
  //gl_Position = P*vec4(pos3D,1);
  //gl_Position = vec4(posHom,-1);
  gl_Position = posOut; //vec4(pos3D,-1);

  //FragColor = vec4(1,0,0,1);
  FragColor = vec4(rgb,1.);
}


