#version 330

uniform sampler2D matcap;
in vec2 vN;

out vec4 color;

void main() {
  vec3 rgb = texture2D(matcap,vN).xyz;
  //color = vec4(vN.x,vN.y,0.,1.);
  color = vec4(rgb,1.);
}
