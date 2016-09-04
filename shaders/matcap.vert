#version 330
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;

uniform sampler2D matcap;
uniform mat4 P;
uniform mat4 MV;

out vec2 vN;

// inspired by https://www.clicktorelease.com/blog/creating-spherical-environment-mapping-shader
void main() {
  // eye direction
  vec3 e = normalize((MV*vec4(pos,1.)).xyz);
  // surface normal direction
  vec3 n = normalize((MV*vec4(normal,0.)).xyz);
  // reflection direction
  vec3 r = reflect(e,n);

  float m = 2. * sqrt( 
      pow( r.x, 2. ) + 
      pow( r.y, 2. ) + 
      pow( r.z + 1., 2. ) 
      );
  vN = r.xy / m + .5;
  gl_Position = P * MV * (vec4(pos,1.));
}
