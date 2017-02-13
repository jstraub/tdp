#version 330

uniform sampler2D labels;
in vec2 uv;

out vec4 color;

void main() {
  vec3 rgb = texture2D(labels,uv).xyz;
  color = vec4(rgb,1.);
}

