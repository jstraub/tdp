#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vColor[];
in vec4 vPosC[];
in vec4 vNC[];
in mat4 vMVP[];
in float vRad[];

out vec2 texcoord;
out vec3 n;
out vec3 v;
out vec3 vColor0;

void main() {
  vColor0 = vColor[0];
  vec3 x = normalize(vec3((vNC[0].y - vNC[0].z), -vNC[0].x, vNC[0].x)) * vRad[0] * 1.41421356;
  vec3 y = cross(vNC[0].xyz, x);

  //n = signMult * vNC[0].xyz;
  n = vNC[0].xyz;

  texcoord = vec2(-1.0, -1.0);
  gl_Position = vMVP[0] * vec4(vPosC[0].xyz + x, 1.0);
  v = vPosC[0].xyz + x;
  EmitVertex();

  texcoord = vec2(1.0, -1.0);
  gl_Position = vMVP[0] * vec4(vPosC[0].xyz + y, 1.0);
  v = vPosC[0].xyz + y;
  EmitVertex();

  texcoord = vec2(-1.0, 1.0);
  gl_Position = vMVP[0] * vec4(vPosC[0].xyz - y, 1.0);
  v = vPosC[0].xyz - y;
  EmitVertex();

  texcoord = vec2(1.0, 1.0);
  gl_Position = vMVP[0] * vec4(vPosC[0].xyz - x, 1.0);
  v = vPosC[0].xyz - x;
  EmitVertex();
  EndPrimitive();

}

