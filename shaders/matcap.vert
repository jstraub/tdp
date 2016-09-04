#version 330

layout (location = 0) in vec3 pos;

uniform mat4 P;
uniform mat4 MV;

out vec4 FragColor;

void main() {
  //gl_Position = P * MV * (vec4(pos,1.));
  gl_Position = P * MV * (vec4(gl_Position.xyz,1.));

  FragColor = vec4(1.,0.,1.,1.);
}
