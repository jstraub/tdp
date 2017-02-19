#version 330

layout (location = 0) in vec3 pos_w;
layout (location = 1) in vec3 n_w;

uniform mat4 P;
uniform mat4 MV;

out vec4 color;

// built in variables overview: 
// https://www.opengl.org/wiki/Built-in_Variable_(GLSL)

void main() {
  gl_Position = P * MV * (vec4(pos_w,1.));
  vec3 n_c = (MV * (vec4(n_w,0.))).xyz;
  if (n_c.x < 0) {
    int id = gl_VertexID;
    float a = 255-(id&0xFF000000)>>24;
    float r = (id&0x00FF0000)>>16;
    float g = (id&0x0000FF00)>>8;
    float b = id&0x000000FF;
    //int r = (id&0xFF000000)>>24;
    //int g = (id&0x00FF0000)>>16;
    //int b = (id&0x0000FF00)>>8;
    //int a = id&0x000000FF;
    color = vec4(r/255., g/255., b/255., 1.);
  } else {
    color = vec4(0,0,0,1);
  }
  //color = vec4(1.,0,0,1);
}


