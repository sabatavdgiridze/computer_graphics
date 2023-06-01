#version 410

layout (location = 0) in vec3 vertex_position;

uniform mat4 projection_matrix, camera_matrix;

void main() {
  gl_Position = projection_matrix * camera_matrix * vec4(vertex_position, 1.0);
}