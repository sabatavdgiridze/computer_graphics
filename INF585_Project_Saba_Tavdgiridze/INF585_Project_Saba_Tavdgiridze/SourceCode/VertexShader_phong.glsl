#version 410

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_normal;

uniform mat4 projection_matrix, camera_matrix;

out vec3 position_camera, normal_camera;

void main() {
    position_camera = vec3(camera_matrix * vec4(vertex_position, 1.0));
    normal_camera = vec3(camera_matrix * vec4(vertex_normal, 0.0));

    gl_Position = projection_matrix * vec4(position_camera, 1.0);
}