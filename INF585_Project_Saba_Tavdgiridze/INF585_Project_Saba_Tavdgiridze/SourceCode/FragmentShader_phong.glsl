#version 410

in vec3 position_camera, normal_camera;

uniform mat4 view_matrix;

struct light_source_str {
	vec3 position;

	vec3 l_s;
	vec3 l_d;
	vec3 l_a;
};

uniform light_source_str light_source;

struct material_str {
	vec3 albedo;

	vec3 m_s;
	vec3 m_d;
	vec3 m_a;
};

uniform material_str material;

float specular_exponent = 100.0;

out vec4 pixel_colour;

void main() {
	vec3 I_a, I_d, I_s;

	I_a = material.m_a * light_source.l_a;

	vec3 n_camera = normalize(normal_camera);
	vec3 light_source_position_camera = vec3(view_matrix * vec4(light_source.position, 1.0));
	vec3 towards_light_source = noramalize(light_source_position_camera - position_camera);

	I_d = material.m_d * light_source.l_d * max(dot(n_camera, towards_light_source), 0.0);

	vec3 towards_viewer = normalize(-position_camera);
	vec3 reflected_towards_viewer = reflect(-towards_light_source, n_camera);
	I_s = material.m_s * light_source.l_s * pow(max(dot(reflected_towards_viewer, towards_light_source), 0.0), specular_exponent);

	pixel_colour = vec4(I_a + I_d + I_s, 1.0);
}