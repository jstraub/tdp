#version 330 core

in vec3 vColor0;
in vec2 texcoord;
in float radius;

out vec4 outColor;

void main() {
  if(dot(texcoord, texcoord) > 1.0f)
    discard;
  outColor = vec4(vColor0, 1.0f);
  gl_FragDepth = gl_FragCoord.z;
}

