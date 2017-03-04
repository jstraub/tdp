#version 330 core
uniform float maxZ;

in vec3 posC;
in vec3 rgbC;
in vec3 nC;
in float rC;
out vec4 outColor;

void main() {
//  outColor = vec4(1,0,0 , 1.);
//  outColor = vec4((gl_FragCoord.x-319.5)/420., (gl_FragCoord.y-239.5)/420., 0,1);
//  gl_FragDepth = ( (posC.z / (2*maxZ)) +0.5f );
//  return;

  vec4 cam = vec4(420., 420., 319.5, 239.5);
//  vec3 ray = normalize(vec3((vec2(gl_FragCoord.x, 480-gl_FragCoord.y) - cam.zw) / cam.xy, 1.f));
  vec3 ray = normalize(vec3((gl_FragCoord.x-319.5)/420., (gl_FragCoord.y-239.5)/420., 1.f));
  vec3 pIntersect = (dot(posC.xyz, nC.xyz) / dot(ray, nC.xyz)) * ray;

  vec3 diff = posC - pIntersect;
//  outColor = vec4(ray.y,ray.y,ray.y , 1.);
//  outColor = vec4(1,0,0 , 1.);
//  return;

  float rSq = pow(rC, 2.);
  if (dot(diff, diff) > rSq) {
//    outColor = vec4(1., rSq/dot(diff,diff), 0., 1.);
//    outColor = vec4(diff, 1.);
    discard;
  } else {
    //outColor = vec4(0., 0., 1., 1.);
    outColor = vec4(rgbC, 1.);
  }
  gl_FragDepth = ( (pIntersect.z / (2*maxZ)) +0.5f );
  gl_FragDepth = ( (posC.z / (2*maxZ)) +0.5f );
}

