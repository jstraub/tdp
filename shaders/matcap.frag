#version 330

uniform sampler2D matcap;
in vec2 vN;

void main() {
  vec3 rgb = texture2D(matcap,vN).xyz;
  //gl_FragColor = vec4(rgb,1.);
  gl_FragColor = vec4(vN.x,vN.y,0.,1.);
}
