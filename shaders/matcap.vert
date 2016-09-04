#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;

uniform sampler2D matcap

uniform mat4 P;
uniform mat4 MV;

out vec4 FragColor;

void main() {
  gl_Position = P * MV * (vec4(pos,1.));
  vec3 n = MV * vec4(normal,1.);
  FragColor = vec4(texture(matcap,n.xy),1.);
}
