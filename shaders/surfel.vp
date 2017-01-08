#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 n;
layout (location = 2) in float r;

uniform mat4 Tinv;
uniform vec4 cam; //cu, cv, fu, fv
uniform float cols;
uniform float rows;
uniform float maxZ;

// Project into the camera plane (into -1 to 1)
vec3 projectCamPlane(vec3 p)
{
  return vec3(((((cam.z * p.x) / p.z) + cam.x) - (cols * 0.5)) / (cols * 0.5),
              ((((cam.w * p.y) / p.z) + cam.y) - (rows * 0.5)) / (rows * 0.5),
              p.z / maxZ);
}
// Project into the image plane (into pixels)
vec3 projectImgPlane(vec3 p)
{
  return vec3(((cam.z * p.x) / p.z) + cam.x,
              ((cam.w * p.y) / p.z) + cam.y,
              p.z);
}

out vec3 posC;
out vec3 nC;
out float rC;
void main() {
  // Transform into camera coordinates
  posC = (Tinv * vec4(pos, 1.0)).xyz;
  nC = normalize(mat3(Tinv) * n.xyz);
  rC = r;

  if (posC.z > maxZ) {
    gl_Position = vec4(1000.f,1000.f,1000.f,1000.f);
    gl_PointSize = 0;
  } else {
    gl_Position = vec4(projectCamPlane(posC), 1.);
    // vectors orthogonal to n
    vec3 u = 1.41421356 * r * vec3(n.y - n.z, -n.x, n.x);
    vec3 v = 1.41421356 * r * cross(n, u);
    // project points on a rectangle around posC into image
    vec3 c1 = projectImgPlane(posC + u);
    vec3 c2 = projectImgPlane(posC + v);
    vec3 c3 = projectImgPlane(posC - u);
    vec3 c4 = projectImgPlane(posC - v);
    // obtain the bounding box corner locations
    vec2 cxs = vec2(min(c1.x, min(c2.x, min(c3.x, c4.x))), max(c1.x, max(c2.x, max(c3.x, c4.x))));
    vec2 cys = vec2(min(c1.y, min(c2.y, min(c3.y, c4.y))), max(c1.y, max(c2.y, max(c3.y, c4.y))));
    // set the size to the maximum side length of the bounding box
    gl_PointSize = max(0, max(abs(cxs.y-cxs.x), abs(cys.y-cys.x))); //r;
  }
}
