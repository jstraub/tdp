#version 330

uniform sampler2D matcap
varying vec2 vN;

void main() {
  gl_FragColor = vec4(texture2D(matcap,vN),1.);
}
