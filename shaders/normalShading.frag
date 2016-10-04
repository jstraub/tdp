#version 400 core

in vec4 FragColor;
in vec3 normal;      //from geometry shader.

out vec4 color;

void main () {
  //gl_FragColor = vec4(normal, 1.0); 
  color = vec4(normal, 1.0); 
  //gl_FragColor = FragColor; 
}
