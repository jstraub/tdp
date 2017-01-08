#version 330 core
in vec3 pos;
out vec4 outColor;
void main() {
  vec2 fpos = gl_PointCoord;
  float d = sqrt(dot(fpos-0.5,fpos-0.5));
  gl_FragColor = vec4(d, 0., 1., 1.);
//  gl_FragDepth =  pos.x;
}

