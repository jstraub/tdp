#version 330

layout (location = 0) in vec3 pos_w;

uniform mat4 T_cw;
uniform vec4 cam; // (fx, fy, uc, vc)
uniform float dMin;
uniform float dMax;
uniform float w;
uniform float h;

out vec4 color;

// built in variables overview: 
// https://www.opengl.org/wiki/Built-in_Variable_(GLSL)

void main() {
  vec4 p_c = T_cw * (vec4(pos_w,1.));
  vec3 x = vec3((2.*(p_c.x/p_c.z * cam.x + cam.z) / w - 1.),
                -(2.*(p_c.y/p_c.z * cam.y + cam.w) / h - 1.),
                (2.*(p_c.z-dMin)/(dMax-dMin) - 1.));
  gl_Position = vec4(x,1);
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
  //color = vec4(1.,0,0,1);
}


