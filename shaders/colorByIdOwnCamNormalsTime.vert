#version 330

layout (location = 0) in vec3 pos_w;
layout (location = 1) in vec3 n_w;
layout (location = 2) in float t;

uniform mat4 T_cw;
uniform vec4 cam; // (fx, fy, uc, vc)
uniform float dMin;
uniform float dMax;
uniform float tMin;
uniform float w;
uniform float h;

out vec4 color;

// built in variables overview: 
// https://www.opengl.org/wiki/Built-in_Variable_(GLSL)

void main() {
  vec3 n_c = (T_cw * (vec4(n_w,0.))).xyz;
  if ( n_c.z < 0 && t > tMin) {
    vec4 p_c = T_cw * (vec4(pos_w,1.));
    vec3 x = vec3((2.*(p_c.x/p_c.z * cam.x + cam.z) / w  - 1.),
        (2.*(p_c.y/p_c.z * cam.y + cam.w) / h - 1.),
        (2.*(p_c.z-dMin)/(dMax-dMin) - 1.));
    gl_Position = vec4(x,1);
    int id = gl_VertexID+1; // 0 is going to be unassigned
    // alpha channel has to be 1 otherwise other colors are rendered
    // scaled by alpha
    float r = (id&0x00FF0000)>>16;
    float g = (id&0x0000FF00)>>8;
    float b = id&0x000000FF;
    //  color = vec4(100./255., 2./255., 1./255., 1.);
    color = vec4(b/255., g/255., r/255., 1.);
  } else {
    gl_Position = vec4(999,999,999,1);
    color = vec4(0,0,0,0);
  }
}


