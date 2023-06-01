#define _USE_MATH_DEFINES
#define NOMINMAX


#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <vector>
#include <stack>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <cmath>

#include <chrono>

#include <glm/glm.hpp>
#include <glm/ext.hpp>


namespace MATH {

    const GLfloat PI = 3.14159265358979323846;

    template<unsigned int dimension>
    std::array<GLfloat, dimension> add(std::array<GLfloat, dimension> first, std::array<GLfloat, dimension> second) {
        std::array<GLfloat, dimension> add;
        for (int idx = 0; idx < dimension; idx++) {
            add.at(idx) = first.at(idx) + second.at(idx);
        }
        return add;
    }
    std::array<GLfloat, 3> add(std::array<GLfloat, 3> first, GLfloat second) {
        std::array<GLfloat, 3> add;
        for (int idx : {0, 1, 2}) {
            add.at(idx) = first.at(idx) + second;
        }
        return add;
    }
    
    std::array<GLfloat, 3> sub(std::array<GLfloat, 3> first, std::array<GLfloat, 3> second) {
        for (int idx : {0, 1, 2}) {
            first.at(idx) = first.at(idx) - second.at(idx);
        }
        return first;
    }
    std::array<GLfloat, 3> sub(std::array<GLfloat, 3> first, GLfloat second) {
        for (int idx : {0, 1, 2}) {
            first.at(idx) = first.at(idx) - second;
        }
        return first;
    }

    GLfloat dot(std::array<GLfloat, 3> first, std::array<GLfloat, 3> second) {
        GLfloat dot_product = 0.0;

        for (int index : {0, 1, 2}) {
            dot_product += first[index] * second[index];
        }

        return dot_product;
    }

    std::array<GLfloat, 3> cross(std::array<GLfloat, 3> first, std::array<GLfloat, 3> second) {
        std::array<GLfloat, 3> cross_product;

        for (int index : {0, 1, 2}) {
            int l_index = (index + 1) % 3;
            int r_index = (index + 2) % 3;

            cross_product[index] = first[l_index] * second[r_index] - first[r_index] * second[l_index];
        }
        return cross_product;
    }

    GLfloat length(std::array<GLfloat, 3> vec) {
        GLfloat length_squared = 0;
        for (int idx : {0, 1, 2}) {
            length_squared = length_squared + vec[idx] * vec[idx];
        }
        return std::sqrt(length_squared);
    }
    template<unsigned int dimension>
    std::array<GLfloat, dimension> multiply(GLfloat multiplier, std::array<GLfloat, dimension> vec) {
        for (int idx = 0; idx < dimension; idx++) {
            vec.at(idx)  = vec.at(idx) * multiplier;
        }
        return vec;
    }

    std::array<GLfloat, 3> min_array(std::array<GLfloat, 3> first, std::array<GLfloat, 3> second) {
        for (int idx : {0, 1, 2}) {
            first[idx] = std::min(first[idx], second[idx]);
        }
        return first;
    }
    std::array<GLfloat, 3> max_array(std::array<GLfloat, 3> first, std::array<GLfloat, 3> second) {
        for (int idx : {0, 1, 2}) {
            first[idx] = std::max(first[idx], second[idx]);
        }
        return first;
    }
};


class Matrix {
    public:
        static Matrix add(Matrix first, Matrix second);
        static Matrix times(GLfloat multiplier, Matrix matrix);
        static Matrix multiply(Matrix first, Matrix second);

        std::array<std::array<GLfloat, 4>, 4> columns;
        

        std::array<GLfloat, 4> multiply(std::array<GLfloat, 4> position) {
            std::array<GLfloat, 4> new_position;
            
            for (int index : {0, 1, 2, 3}) {
                new_position[index] = 0.0;
            }
            
            for (int index : {0, 1, 2, 3}) {
                new_position = MATH::add(new_position, MATH::multiply(position[index], columns[index]));
            }
            return new_position;
        }

        Matrix transpose() {
            std::array<std::array<GLfloat, 4>, 4> new_columns;
            for (int first_index = 0; first_index < 4; first_index++) {
                for (int second_index = 0; second_index < 4; second_index++) {
                    new_columns[first_index][second_index] = columns[second_index][first_index];
                }
            }
            Matrix result(new_columns);
            return result;
        }

        Matrix(std::array<std::array<GLfloat, 4>, 4> columns) {
            this->columns = columns;
        }
        Matrix() {
            
        }
    private:

};

Matrix Matrix::add(Matrix first, Matrix second) {
    std::array<std::array<GLfloat, 4>, 4> columns;

    for (int index : {0, 1, 2, 3}) {
        columns[index] = MATH::add(first.columns[index], second.columns[index]);
    }
    Matrix result(columns);
    return result;
}

Matrix Matrix::times(GLfloat multiplier, Matrix matrix) {
    for (auto& column : matrix.columns) {
        column = MATH::multiply(multiplier, column);
    }
    return matrix;
}

Matrix Matrix::multiply(Matrix first, Matrix second) {
    std::array<std::array<GLfloat, 4>, 4> columns;

    for (int index : {0, 1, 2, 3}) {
        columns[index] = first.multiply(second.columns[index]);
    }
    Matrix result(columns);
    return result;
}


class Rotation {
    public:
        std::array<GLfloat, 3> axis;
        GLfloat theta;

        Rotation(GLfloat theta, std::array<GLfloat, 3> axis) {
            this->theta = theta;
            this->axis = axis;
        }

        std::array<GLfloat, 3> rotate(std::array<GLfloat, 3> position) {
            std::array<GLfloat, 3> new_position;
            new_position = MATH::add(MATH::multiply(cos(theta), position), MATH::add(MATH::multiply(sin(theta), MATH::cross(axis, position)), MATH::multiply((1 - cos(theta)) * MATH::dot(axis, position), axis)));
            return new_position;
        }
    private:

};

class BoundingBox;

class Ball {
    public:

        static bool intersect(std::shared_ptr<Ball> first, std::shared_ptr<Ball> second);

        std::array<GLfloat, 3> position;
        std::array<GLfloat, 3> velocity;
        std::array<GLfloat, 3> force;

        GLfloat mass;
        GLfloat radius;

        Ball(GLfloat radius, GLfloat mass, std::array<GLfloat, 3> position, std::array<GLfloat, 3> velocity) {
            this->radius = radius;
            this->mass = mass;

            this->position = position;
            this->velocity = velocity;
        }

        std::shared_ptr<BoundingBox> create_BoundingBox() {
            std::shared_ptr<BoundingBox> bounding_box = std::make_shared<BoundingBox>(MATH::sub(this->position, this->radius), MATH::add(this->position, this->radius));
            return bounding_box;
        }

    private:
};


bool Ball::intersect(std::shared_ptr<Ball> first, std::shared_ptr<Ball> second) {
    return MATH::length(MATH::sub(first->position, second->position)) < (first->radius + second->radius);
}

class BoundingBox {
    public:
        static bool intersect(std::shared_ptr<BoundingBox> first, std::shared_ptr<BoundingBox> second);
        static std::shared_ptr<BoundingBox> union_BoundingBox(std::shared_ptr<BoundingBox> first, std::shared_ptr<BoundingBox> second);

        std::array<GLfloat, 3> min_axis;
        std::array<GLfloat, 3> max_axis;

        BoundingBox(std::array<GLfloat, 3> min_axis, std::array<GLfloat, 3> max_axis) {
            this->min_axis = min_axis;
            this->max_axis = max_axis;
        }
    private:

};

bool BoundingBox::intersect(std::shared_ptr<BoundingBox> first, std::shared_ptr<BoundingBox> second) {
    for (int dimension : {0, 1, 2}) {
        if (first->min_axis[dimension] > second->max_axis[dimension]) {
            return false;
        }
        if (first->max_axis[dimension] < second->min_axis[dimension]) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<BoundingBox> BoundingBox::union_BoundingBox(std::shared_ptr<BoundingBox> first, std::shared_ptr<BoundingBox> second) {
    return std::make_shared<BoundingBox>(MATH::min_array(first->min_axis, second->min_axis), MATH::max_array(first->max_axis, second->max_axis));
}




class BVH_node {
    public:
        std::shared_ptr<BoundingBox> bounding_box;

        std::shared_ptr<BVH_node> first_child;
        std::shared_ptr<BVH_node> second_child;

        std::shared_ptr<Ball> ptr_ball;

        BVH_node(std::shared_ptr<BoundingBox> bounding_box, std::shared_ptr<BVH_node> first_child, std::shared_ptr<BVH_node> second_child, std::shared_ptr<Ball> ptr_ball) {
            this->bounding_box = bounding_box;

            this->first_child = first_child;
            this->second_child = second_child;

            this->ptr_ball = ptr_ball;
        }

    private:

};


class BVH {
    public:
        static std::shared_ptr<BVH_node> create_BVH_node(std::vector<std::shared_ptr<Ball>> balls);


        std::shared_ptr<BVH_node> root;

        BVH(std::shared_ptr<BVH_node> root) {
            this->root = root;
        }
        BVH(std::vector<std::shared_ptr<Ball>> balls) {
            this->root = BVH::create_BVH_node(balls);
        }
        void intersections(std::vector<std::shared_ptr<Ball>> &intersections_vector, std::shared_ptr<Ball> ptr_ball) {
            std::stack<std::shared_ptr<BVH_node>> unexplored;
            unexplored.push(root);
            
            int nodes_considered = 0;

            while(!unexplored.empty()) {
                nodes_considered++;
                auto current_node = unexplored.top();
                unexplored.pop();

                if (current_node->ptr_ball != nullptr) {
                    if ((current_node->ptr_ball != ptr_ball) && Ball::intersect(ptr_ball, current_node->ptr_ball)) {
                        intersections_vector.push_back(current_node->ptr_ball);
                    }
                } else {
                    if (BoundingBox::intersect(current_node->first_child->bounding_box, ptr_ball->create_BoundingBox())) {
                        unexplored.push(current_node->first_child);
                    }
                    if (BoundingBox::intersect(current_node->second_child->bounding_box, ptr_ball->create_BoundingBox())) {
                        unexplored.push(current_node->second_child);
                    }
                }

            }
            // std::cout << "BVH considered " << nodes_considered << " number of nodes" << std::endl;

        }

    private:

};



std::shared_ptr<BVH_node> BVH::create_BVH_node(std::vector<std::shared_ptr<Ball>> balls) {
    if (balls.size() == 1) {
        std::shared_ptr<BVH_node> root = std::make_shared<BVH_node>(balls[0]->create_BoundingBox(), nullptr, nullptr, balls[0]);
        return root;
    } else {
        
        auto variance_along_dimension = [&balls] (int dimension) {
            GLfloat min_dimension = 0;
            GLfloat max_dimension = 0;

            for (auto ptr_ball : balls) {
                min_dimension = std::min(min_dimension, ptr_ball->position[dimension]);
                max_dimension = std::max(max_dimension, ptr_ball->position[dimension]);
            }
            return (max_dimension - min_dimension);
        };
        std::vector<GLfloat> variances;
        for (int dimension : {0, 1, 2}) {
            variances.push_back(variance_along_dimension(dimension));
            // std::cout << ". " << variance_along_dimension(dimension) << std::endl;
        }

        int dimension = std::distance(variances.begin(), std::max_element(variances.begin(), variances.end()));

        // std::cout << "... " << dimension << std::endl;

        int length = balls.size();
        std::nth_element(balls.begin(), std::next(balls.begin(), length / 2), balls.end(), [dimension](const std::shared_ptr<Ball> first, const std::shared_ptr<Ball> second) {
            return (first->position[dimension]) < (second->position[dimension]);
        });

        std::shared_ptr<BVH_node> first_child = create_BVH_node(std::vector<std::shared_ptr<Ball>>(balls.begin(), std::next(balls.begin(), length / 2)));
        std::shared_ptr<BVH_node> second_child = create_BVH_node(std::vector<std::shared_ptr<Ball>>(std::next(balls.begin(), length / 2), balls.end()));

        std::shared_ptr<BVH_node> root = std::make_shared<BVH_node>(BoundingBox::union_BoundingBox(first_child->bounding_box, second_child->bounding_box), first_child, second_child, nullptr);
        return root;
    }
}


class Triangle {
    public:
        std::array<std::array<GLfloat, 3>, 3> positions;

        Triangle(std::array<std::array<GLfloat, 3>, 3> positions) {
            this->positions = positions;
        }
    private:
};

std::vector<Triangle> polygon_to_triangles(std::vector<std::array<GLfloat, 3>> points) {
    std::vector<Triangle> triangles;
    int n_gon = points.size();

    for (int index = 1; index < n_gon - 1; index++) {
        std::array<std::array<GLfloat, 3>, 3> triangle_points;
        
        triangle_points[0] = points[0];
        triangle_points[1] = points[index];
        triangle_points[2] = points[index + 1];

        triangles.push_back(Triangle(triangle_points));
    }

    return triangles;
}

std::array<GLfloat, 3> point_on_sphere(std::array<GLfloat, 3> origin, GLfloat radius, GLfloat theta_angle, GLfloat phi_angle) {
    std::array<GLfloat, 3> point_coordinate = origin;
    

    point_coordinate[0] += radius * sin(phi_angle) * cos(theta_angle);
    point_coordinate[1] += radius * sin(phi_angle) * sin(theta_angle);
    point_coordinate[2] += radius * cos(phi_angle);

    return point_coordinate;
}



std::pair<std::vector<Triangle>, std::vector<Triangle>> triangle_mesh_sphere(std::array<GLfloat, 3> origin, GLfloat radius) {
    int theta_count = 16;
    int phi_count = 8;
    
    auto convert_to_theta_angle = [theta_count] (int theta) {
        return (GLfloat) ((2.0 * MATH::PI * theta) / theta_count);
    };
    auto convert_to_phi_angle = [phi_count] (int phi) {
        return (GLfloat) ((MATH::PI * phi) / phi_count - MATH::PI / 2.0);
    };

    std::vector<Triangle> triangles;
    std::vector<Triangle> normals;

    for (int theta = 0; theta < theta_count; theta++) {
        for (int phi = 0; phi < phi_count; phi++) {

            if (phi == 0 || phi == phi_count - 1) {
                if (phi == 0) {
                    std::vector<std::pair<int, int>> indexes;

                    indexes.push_back(std::make_pair(theta, phi));
                    indexes.push_back(std::make_pair(theta + 1, phi + 1));
                    indexes.push_back(std::make_pair(theta, phi + 1));

                    std::array<std::array<GLfloat, 3>, 3> triangle_points;
                    std::array<std::array<GLfloat, 3>, 3> normal_points;

                    for (int index = 0; index < 3; index++) {
                        triangle_points[index] = point_on_sphere(origin, radius, convert_to_theta_angle(indexes[index].first), convert_to_phi_angle(indexes[index].second));
                        normal_points[index] = point_on_sphere(std::array<GLfloat, 3> {0.0, 0.0, 0.0}, radius, convert_to_theta_angle(indexes[index].first), convert_to_phi_angle(indexes[index].second));
                    }
                    triangles.push_back(Triangle(triangle_points));
                    normals.push_back(Triangle(normal_points));
                }
                if (phi == phi_count - 1) {
                    std::vector<std::pair<int, int>> indexes;

                    indexes.push_back(std::make_pair(theta, phi));
                    indexes.push_back(std::make_pair(theta + 1, phi));
                    indexes.push_back(std::make_pair(theta, phi + 1));

                    std::array<std::array<GLfloat, 3>, 3> triangle_points;
                    std::array<std::array<GLfloat, 3>, 3> normal_points;

                    for (int index = 0; index < 3; index++) {
                        triangle_points[index] = point_on_sphere(origin, radius, convert_to_theta_angle(indexes[index].first), convert_to_phi_angle(indexes[index].second));
                        normal_points[index] = point_on_sphere(std::array<GLfloat, 3> {0.0, 0.0, 0.0}, radius, convert_to_theta_angle(indexes[index].first), convert_to_phi_angle(indexes[index].second));
                    }
                    triangles.push_back(Triangle(triangle_points));
                    normals.push_back(Triangle(normal_points));
                }
            } else {
                std::vector<std::pair<int, int>> indexes;

                indexes.push_back(std::make_pair(theta, phi));
                indexes.push_back(std::make_pair(theta + 1, phi));
                indexes.push_back(std::make_pair(theta + 1, phi + 1));
                indexes.push_back(std::make_pair(theta, phi + 1));

                std::vector<std::array<GLfloat, 3>> rectangle_points;

                for (auto pair : indexes) {
                    rectangle_points.push_back(point_on_sphere(origin, radius, convert_to_theta_angle(pair.first), convert_to_phi_angle(pair.second)));
                }
                
                auto rectangle_triangles = polygon_to_triangles(rectangle_points);
                for (auto triangle : rectangle_triangles) {
                    triangles.push_back(triangle);

                    auto normal_triangle = triangle;
                    for (auto& point : normal_triangle.positions) {
                        point = MATH::sub(point, origin);
                    }
                    normals.push_back(normal_triangle);
                }

            }
        }
    }
    return std::make_pair(triangles, normals);
}

class GPU_DATA {
    public:
        std::vector<std::array<GLfloat, 3>> points;
        std::vector<int> triangle_indices;

        GPU_DATA(std::vector<std::array<GLfloat, 3>> points, std::vector<int> triangle_indices) {
            this->points = points;
            this->triangle_indices = triangle_indices;
        }
    private:

};


GPU_DATA towards_gpu(std::vector<Triangle> triangles) {
    std::set<std::array<GLfloat, 3>> points;
    std::vector<int> triangles_indices;
    for (auto triangle : triangles) {
        for (int index : {0, 1, 2}) {
            points.insert(triangle.positions[index]);
            int position_index = std::distance(points.begin(), points.find(triangle.positions[index]));
            triangles_indices.push_back(position_index);
        }
    }
    std::vector<std::array<GLfloat, 3>> vector_points(points.begin(), points.end());

    return GPU_DATA(vector_points, triangles_indices);
}


class Plane {
    public:
        std::array<GLfloat, 3> origin;
        std::array<GLfloat, 3> normal;

        Plane(std::array<GLfloat, 3> origin, std::array<GLfloat, 3> normal) {
            this->origin = origin;
            this->normal = normal;
        }

        GLfloat distance(std::array<GLfloat, 3> point) {
            return MATH::dot(normal, MATH::sub(point, origin));
        }

        void resolve_collision(std::shared_ptr<Ball> ball) {
            if (distance(ball->position) < ball->radius) {
                GLfloat penetration_distance = ball->radius - distance(ball->position);
                
                std::array<GLfloat, 3> velocity_orthogonal = MATH::multiply(MATH::dot(ball->velocity, normal),normal);
                std::array<GLfloat, 3> velocity_tangential = MATH::sub(ball->velocity, velocity_orthogonal);

                GLfloat alpha = 1.0;
                GLfloat beta = 1.0;

                ball->velocity = MATH::sub(MATH::multiply(alpha, velocity_tangential), MATH::multiply(beta, velocity_orthogonal));
                ball->position = MATH::add(ball->position, MATH::multiply(penetration_distance, normal));
            }
        }

    private:

};

class Gravity {
    public:
        std::array<GLfloat, 3> gravity_force;

				Gravity() {
					this->gravity_force = std::array<GLfloat, 3> {0.0, 0.0, -10.0};
				}
        Gravity(std::array<GLfloat, 3> gravity_force) {
            this->gravity_force = gravity_force;
        }

        void apply_force(std::shared_ptr<Ball> ball) {
            ball->force = MATH::add(ball->force, gravity_force);
        }
    private:
};

void resolve_collision(std::shared_ptr<Ball> first_ball, std::shared_ptr<Ball> second_ball) {
    auto u_contact = MATH::multiply(1.0 / MATH::length(MATH::sub(first_ball->position, second_ball->position)), MATH::sub(first_ball->position, second_ball->position));
    
    GLfloat centers_distance = MATH::length(MATH::sub(first_ball->position, second_ball->position));
    GLfloat penetration_distance = (first_ball->radius + second_ball->radius) - centers_distance;

    first_ball->position = MATH::add(first_ball->position, MATH::multiply(penetration_distance / 2.0, u_contact));
    second_ball->position = MATH::sub(second_ball->position, MATH::multiply(penetration_distance / 2.0, u_contact));

    first_ball->velocity = MATH::add(first_ball->velocity, MATH::multiply(2 * (second_ball->mass / (first_ball->mass + second_ball->mass)) * MATH::dot(MATH::sub(first_ball->velocity, second_ball->velocity), u_contact), u_contact));
    second_ball->velocity = MATH::sub(second_ball->velocity, MATH::multiply(2 * (first_ball->mass / (first_ball->mass + second_ball->mass)) * MATH::dot(MATH::sub(first_ball->velocity, second_ball->velocity), u_contact), u_contact));

}


class collision_resolution {
    public:
        virtual void resolve_collisions(std::vector<std::shared_ptr<Ball>>& balls) const = 0;
    private:
};

class naive_collision_resolution : public collision_resolution {
    public:
        virtual void resolve_collisions(std::vector<std::shared_ptr<Ball>>& balls) const override;
    private:

};

void naive_collision_resolution::resolve_collisions(std::vector<std::shared_ptr<Ball>>& balls) const {
    for (auto ball_ptr : balls) {
        for (auto other_ball_ptr : balls) {
            if (ball_ptr != other_ball_ptr) {
                resolve_collision(ball_ptr, other_ball_ptr);
            }
        }
    }
}


class BVH_collision_resolution : public collision_resolution {
    public:
        virtual void resolve_collisions(std::vector<std::shared_ptr<Ball>>& balls) const override;
    private:
};


void BVH_collision_resolution::resolve_collisions(std::vector<std::shared_ptr<Ball>>& balls) const {
    BVH BVH_structure(balls);

    for (auto ball_ptr : balls) {
        std::vector<std::shared_ptr<Ball>> intersections;
        BVH_structure.intersections(intersections, ball_ptr);

        for (auto other_ball_ptr : intersections) {
            resolve_collision(ball_ptr, other_ball_ptr);
        }
    }
}

class Scene {
    public:
        std::vector<std::shared_ptr<Ball>> balls;
        std::vector<Plane> planes;

        Gravity gravity;
        std::shared_ptr<collision_resolution> collision_resolver;

				Scene() {
					this->gravity = Gravity();
					this->collision_resolver = std::make_shared<naive_collision_resolution>();
				}

        Scene(std::vector<Plane> planes, Gravity gravity, std::shared_ptr<collision_resolution> collision_resolver) {
            this->planes = planes;

            this->gravity = gravity;
            this->collision_resolver = collision_resolver;
        }

        void add_ball_ptr(std::shared_ptr<Ball> ball_ptr) {
            balls.push_back(ball_ptr);
        }

        void resolve_collision() {
            collision_resolver->resolve_collisions(balls);
        }

        void advance(GLfloat time_interval) {
            // force calculations
            for (auto ball_ptr : balls) {
                ball_ptr->force = std::array<GLfloat, 3> {0.0, 0.0, 0.0};
            }
            for (auto ball_ptr : balls) {
                gravity.apply_force(ball_ptr);
            }
            for (auto ball_ptr : balls) {
                ball_ptr->velocity  = MATH::add(ball_ptr->velocity, MATH::multiply(time_interval, MATH::multiply(1.0 / ball_ptr->mass, ball_ptr->force)));
                ball_ptr->position = MATH::add(ball_ptr->position, MATH::multiply(time_interval, ball_ptr->velocity));
            }
            // impulse calculations
            for (auto plane : planes) {
                for (auto ball_ptr : balls) {
                    plane.resolve_collision(ball_ptr);
                }
            }
            resolve_collision();
        }

    private:

};

class Simulation {
    public:
        GLfloat initial_time;
        GLfloat simulation_time;

        GLfloat max_time_interval;
        Scene scene;

        Simulation(GLfloat max_time_interval, Scene scene) {
            this->max_time_interval = max_time_interval;
            this->scene = scene;
        }
        void init(GLfloat initial_time) {
            this->initial_time = initial_time;
            this->simulation_time = simulation_time;
        }
        void simulate(GLfloat current_time) {
            while (simulation_time < current_time) {
                if (simulation_time + max_time_interval < current_time) {
                    scene.advance(max_time_interval);
                    simulation_time += max_time_interval;
                } else {
                    scene.advance(current_time - simulation_time);
                    simulation_time = current_time;
                }
            }
        }
    private:
};

class Camera {
	public:
		Camera(std::array<GLfloat, 3> camera_position, std::array<GLfloat, 3> look_at_direction, std::array<GLfloat, 3> pre_up_vector) {
            std::array<GLfloat, 3> right = MATH::cross(look_at_direction, pre_up_vector);
            right = MATH::multiply(1.0 / MATH::length(right), right);

            std::array<GLfloat, 3> up_vector = MATH::cross(right, look_at_direction);

            std::array<std::array<GLfloat, 4>, 4> columns_camera_to_world;
            std::array<std::array<GLfloat, 4>, 4> columns_world_to_camera;

            columns_camera_to_world[0] = extend(right, 0.0);
            columns_camera_to_world[1] = extend(up_vector, 0.0);
            columns_camera_to_world[2] = extend(MATH::multiply(-1.0, look_at_direction), 0.0);
            columns_camera_to_world[3] = extend(camera_position, 1.0);
            camera_to_world = Matrix(columns_camera_to_world);

            std::array<std::array<GLfloat, 3>, 3> frame_matrix = transpose(std::array<std::array<GLfloat, 3>, 3> {right, up_vector, MATH::multiply(-1.0, look_at_direction)});

            /**
            
            std::cout << "Frame Matrix" << std::endl;
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    std::cout << frame_matrix[y][x] << " "; 
                }
                std::cout << std::endl;
            }

            **/

            columns_world_to_camera[0] = extend(frame_matrix[0], 0.0);
            columns_world_to_camera[1] = extend(frame_matrix[1], 0.0);
            columns_world_to_camera[2] = extend(frame_matrix[2], 0.0);

            std::array<GLfloat, 3> fourth_column;
            fourth_column = MATH::add(MATH::add(MATH::multiply(camera_position[2], frame_matrix[2]), MATH::multiply(camera_position[1], frame_matrix[1])), MATH::multiply(camera_position[0], frame_matrix[0]));
            fourth_column = MATH::multiply(-1.0, fourth_column);
            
            /**

            for (int x = 0; x < 3; x++) {
                std::cout << " -> " << fourth_column[x] << std::endl;
            }

            **/

            columns_world_to_camera[3] = extend(fourth_column, 1.0);

            world_to_camera = Matrix(columns_world_to_camera);

		}
        std::vector<GLfloat> column_major() {
            std::vector<GLfloat> entities;
            for (int column : {0, 1, 2, 3}) {
                for (auto entity : world_to_camera.columns.at(column)) {
                    entities.push_back(entity);
                }
            }
            
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    std::cout << camera_to_world.columns[y][x] << " ";
                }
                std::cout << std::endl;
            }

            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    std::cout << world_to_camera.columns[y][x] << " ";
                }
                std::cout << std::endl;
            }

            return entities;
        }
        std::vector<GLfloat> projection_matrix(GLfloat z_near, GLfloat z_far, GLfloat theta) {

            std::array<std::array<GLfloat, 4>, 4> projection_matrix;

            GLfloat f = 1.0 / tan(theta / 2);
            GLfloat L = z_near - z_far;
            GLfloat C = (z_near + z_far) / L;
            GLfloat D = 2 * z_near * z_far / L;

            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    projection_matrix[y][x] = 0.0;
                }
            }

            projection_matrix[0][0] = f;
            projection_matrix[1][1] = f;

            projection_matrix[2][2] = C;
            projection_matrix[3][2] = D;

            projection_matrix[2][3] = -1.0f;

            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    std::cout << projection_matrix[y][x] << " ";
                }
                std::cout << std::endl;
            }

            std::vector<GLfloat> flattened;

            for (int first_index : {0, 1, 2, 3}) {
                for (int second_index : {0, 1, 2, 3}) {
                    flattened.push_back(projection_matrix[first_index][second_index]);
                }
            }

            return flattened;
        }

	private:
        std::array<GLfloat, 4> extend(std::array<GLfloat, 3> input_array, GLfloat additional_value) {
            std::array<GLfloat, 4> result;
            
            for (int index : {0, 1, 2}) {
                result[index] = input_array[index];
            }
            result[3] = additional_value;

            return result;
        }
        template<unsigned int dimension>
        std::array<std::array<GLfloat, dimension>, dimension> transpose(std::array<std::array<GLfloat, dimension>, dimension> input_matrix) {
            std::array<std::array<GLfloat, dimension>, dimension> output_matrix;
            for (int first_index = 0; first_index < dimension; first_index++) {
                for (int second_index = 0; second_index < dimension; second_index++) {
                    output_matrix[first_index][second_index] = input_matrix[second_index][first_index];
                }
            }
            return output_matrix;
        }



        Matrix camera_to_world;
        Matrix world_to_camera;
};

class Graphics {
    public:

        void show_wireframe(GLFWwindow * window, GLuint opengl_program, std::vector<std::array<std::array<GLfloat, 3>, 2>> lines) {
            int num_lines = lines.size();
            GLfloat * lines_array = (GLfloat *) malloc(6 * sizeof(GLfloat) * num_lines);

            for (int line_index = 0; line_index < num_lines; line_index++) {
                for (int point_index : {0, 1}) {
                    for (int coordinate_index : {0, 1, 2}) {
                        lines_array[6 * line_index + 3 * point_index + coordinate_index] = lines[line_index][point_index][coordinate_index];
                    }
                }
            }

            for (int idx = 0; idx < 6 * num_lines; idx++) {
                std::cout << lines_array[idx] << " ";   
            }
            std::cout << std::endl;

            GLuint points_vbo;
            
            glGenBuffers(1, &points_vbo);
            glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
            glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(GLfloat) * num_lines, lines_array, GL_STATIC_DRAW);

            GLuint vertex_array_object;

            glGenVertexArrays(1, &vertex_array_object);
            glBindVertexArray(vertex_array_object);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(0);

            glUseProgram(opengl_program);
            glBindVertexArray(vertex_array_object);
            glDrawArrays(GL_LINES, 0, 2 * num_lines);

            glDeleteVertexArrays(1, &vertex_array_object);
            glDeleteBuffers(1, &points_vbo);

            free(lines_array);
        }

        void Show_triangles(GLFWwindow * window, GLuint opengl_program, std::vector<Triangle> triangles, std::vector<Triangle> normals) {
            int num_triangles = triangles.size();
            GLfloat * triangles_array = (GLfloat *) malloc(9 * sizeof(GLfloat) * num_triangles);
            GLfloat * normals_array = (GLfloat *) malloc(9 * sizeof(GLfloat) * num_triangles);

            for (int triangle_index = 0; triangle_index < num_triangles; triangle_index++) {
                for (int point_index : {0, 1, 2}) {
                    for (int coordinate_index : {0, 1, 2}) {
                        triangles_array[9 * triangle_index + 3 * point_index + coordinate_index] = triangles[triangle_index].positions[point_index].at(coordinate_index);
                        normals_array[9 * triangle_index + 3 * point_index + coordinate_index] = normals[triangle_index].positions[point_index].at(coordinate_index);
                    }
                }
            }

            GLuint points_vbo;
            GLuint normals_vbo;

            glGenBuffers(1, &points_vbo);
            glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
            glBufferData(GL_ARRAY_BUFFER, num_triangles * 9 * sizeof(GLfloat), triangles_array, GL_STATIC_DRAW);

            glGenBuffers(1, &normals_vbo);
            glBindBuffer(GL_ARRAY_BUFFER, normals_vbo);
            glBufferData(GL_ARRAY_BUFFER, num_triangles * 9 * sizeof(GLfloat), normals_array, GL_STATIC_DRAW);

            GLuint vertex_array_object;
            
            glGenVertexArrays(1, &vertex_array_object);
            glBindVertexArray(vertex_array_object);

            glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

            glBindBuffer(GL_ARRAY_BUFFER, normals_vbo);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);

            glUseProgram(opengl_program);
            glBindVertexArray(vertex_array_object);
            glDrawArrays(GL_TRIANGLES, 0, 3 * num_triangles);



            glDeleteVertexArrays(1, &vertex_array_object);
            glDeleteBuffers(1, &points_vbo);
            glDeleteBuffers(1, &normals_vbo);

            free(triangles_array);
            free(normals_array);
        }


    private:
};


class Drawer {
    public:
			Drawer(Camera camera_arg) : camera(camera_arg) {
				window = nullptr;
			}

			void Init(std::string vertex_shader_file, std::string fragment_shader_file) {
				init_glfw();
				init_opengl();
                init_gpu(vertex_shader_file, fragment_shader_file);
			}
            void clear() {
                glDeleteProgram(opengl_program);
                glfwDestroyWindow(window);
                glfwTerminate();
            }

			GLFWwindow * window;
			GLuint opengl_program;
            Graphics drawer_graphics;

			Camera camera;

			std::string file_to_string(std::string file_name) {
				std::ifstream stream_reader(file_name.c_str());
				std::stringstream string_buffer;

				string_buffer << stream_reader.rdbuf();
				return string_buffer.str();
			}
			void load_shader(GLuint opengl_program, GLenum shader_type, std::string shader_file_name) {
				GLuint shader = glCreateShader(shader_type);
				std::string shader_source_string =	file_to_string(shader_file_name);

				GLchar * shader_source = (GLchar *) shader_source_string.c_str();

				glShaderSource(shader, 1, &shader_source, NULL);
				glCompileShader(shader);

				glAttachShader(opengl_program, shader);
				glDeleteShader(shader);
			}

			static void window_resize_callback(GLFWwindow * window, int width, int height) {
				glViewport(0, 0, (GLint)width, (GLint)height);
			}

			void init_glfw() {
				if (!glfwInit()) {
					std::cerr << "ERROR: Failed to initialize GLFW" << std::endl;
					std::exit(EXIT_FAILURE);
				}

				glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
				glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
				glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
				glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

				window = glfwCreateWindow(1024, 768, "INF 585 - Project Saba Tavdgiridze", nullptr, nullptr);
				if (!window) {
					std::cerr << "ERROR: Failed to open window" << std::endl;
					glfwTerminate();
					std::exit(EXIT_FAILURE);
				}

				glfwMakeContextCurrent(window);
				glfwSetWindowSizeCallback(window, window_resize_callback);
			}

			void init_opengl() {
				if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
					std::cerr << "Error: Failed to initialize OpenGL context" << std::endl;
					glfwTerminate();
					std::exit(EXIT_FAILURE);
				}

                /**
                
                glEnable(GL_CULL_FACE);
				glCullFace(GL_BACK);
				glFrontFace(GL_CCW);
                
                **/

				glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);

				glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			}

            void init_gpu(std::string vertex_shader_file, std::string fragment_shader_file) {
                opengl_program = glCreateProgram();
                load_shader(opengl_program, GL_VERTEX_SHADER, vertex_shader_file);
                load_shader(opengl_program, GL_FRAGMENT_SHADER, fragment_shader_file);
                glLinkProgram(opengl_program);
            }
    private:

};

class Application {
    public:

        Application(Camera application_camera) : application_drawer(Drawer(application_camera)) {

        }

        void run() {

        }
    private:
        Drawer application_drawer;
};

// For testing purposes [Begin]

// checking BoundingBox class [Begin]

void application_1(int argc, char ** argv) {
    while (true) {
        std::array<GLfloat, 3> min_axis_first;
        std::array<GLfloat, 3> max_axis_first;

        for (int index : {0, 1, 2}) {
            std::cin>>min_axis_first[index];
        }
        for (int index : {0, 1, 2}) {
            std::cin>>max_axis_first[index];
        }
        std::shared_ptr<BoundingBox> bounding_box_first = std::make_shared<BoundingBox>(min_axis_first, max_axis_first);
        

        std::array<GLfloat, 3> min_axis_second;
        std::array<GLfloat, 3> max_axis_second;

        for (int index : {0, 1, 2}) {
            std::cin>>min_axis_second[index];
        }
        for (int index : {0, 1, 2}) {
            std::cin>>max_axis_second[index];
        }
        std::shared_ptr<BoundingBox> bounding_box_second = std::make_shared<BoundingBox>(min_axis_second, max_axis_second);
        if (BoundingBox::intersect(bounding_box_first, bounding_box_second)) {
            std::cout << "They intersect!" << std::endl;
        } else {
            std::cout << "They don't intersect!" << std::endl;
        }
        auto union_bounding_box = BoundingBox::union_BoundingBox(bounding_box_first, bounding_box_second);

        std::cout << "Min axis :" << std::endl;
        for (int index : {0, 1, 2}) {
            std::cout << union_bounding_box->min_axis[index] << " ";
        }
        std::cout << std::endl;

        std::cout << "Max axis :" << std::endl;
        for (int index : {0, 1, 2}) {
            std::cout << union_bounding_box->max_axis[index] << " ";
        }
        std::cout << std::endl;


    }
}

// cheking BoundingBox class [End]

// checking BVH class [Begin]

void application_2(int argc, char ** argv) {
    int N_points = atoi(argv[1]);
    int Seed = atoi(argv[2]);

    GLfloat bounding_box_edge_length = atof(argv[3]);

    std::vector<std::shared_ptr<Ball>> balls;
    
    srand(Seed);
    for (int index = 0; index < N_points; index++) {
        std::array<GLfloat, 3> position;
        for (int coordinate_index : {0, 1, 2}) {
            position[coordinate_index] = ((GLfloat) rand() / RAND_MAX) * bounding_box_edge_length;
        }

        /**
        GLfloat min_scale = 1.0 / 50.0;
        GLfloat max_scale = 1.0 / 10.0;

        GLfloat radius = (min_scale * bounding_box_edge_length) + ((GLfloat) rand() / RAND_MAX) * ((max_scale - min_scale) * bounding_box_edge_length);
        **/

        GLfloat radius = 5.0;
        GLfloat mass_constant = 1.0;

        balls.push_back(std::make_shared<Ball>(radius, mass_constant, position, std::array<GLfloat, 3> {0.0, 0.0, 0.0}));

    }
    std::cout << "random balls have been generated!" << std::endl;

    std::shared_ptr<BVH> BVH_ptr = std::make_shared<BVH>(balls);
    std::cout << "BVH hierarchy has been constructed" << std::endl;

    std::vector<std::shared_ptr<Ball>> intersections_BVH;
    std::vector<std::shared_ptr<Ball>> intersections_naive;

    int max_intersection_size = 0;

    for (auto ball_ptr : balls) {
        intersections_BVH.clear();
        intersections_naive.clear();

        BVH_ptr->intersections(intersections_BVH, ball_ptr);

        for (auto other_ball_ptr : balls) {
            if (ball_ptr != other_ball_ptr) {
                if (Ball::intersect(ball_ptr, other_ball_ptr)) {
                    intersections_naive.push_back(other_ball_ptr);
                }
            }
        }
        
        if (intersections_BVH.size() != intersections_naive.size()) {
            std::cout << "The BVH implemntation is wrong! (size mismatch)" << std::endl;
            std::cout << intersections_BVH.size() << " " << intersections_naive.size() << std::endl; 
            return;
        }
        for (auto element_BVH : intersections_BVH) {
            if (std::find(intersections_naive.begin(), intersections_naive.end(), element_BVH) == intersections_naive.end()) {
                std::cout << "The BVH implemntation is wrong!" << std::endl;
                return;
            }
        }
        for (auto element_naive : intersections_naive) {
            if (std::find(intersections_BVH.begin(), intersections_BVH.end(), element_naive) == intersections_BVH.end()) {
                std::cout << "The BVH implemntation is wrong!" << std::endl;
                return;
            }
        }
        max_intersection_size = std::max(max_intersection_size, (int)intersections_BVH.size());
    }
    std::cout << "The BVH implementation seems to be working!" << std::endl;
    std::cout << "The max intersection size = " << max_intersection_size << std::endl;


    auto start_BVH = std::chrono::steady_clock::now();
    for (auto ball_ptr : balls) {
        intersections_BVH.clear();
        BVH_ptr->intersections(intersections_BVH, ball_ptr);
    }
    auto finish_BVH = std::chrono::steady_clock::now();
    std::cout << "BVH took " << std::chrono::duration_cast<std::chrono::milliseconds>(finish_BVH - start_BVH).count() << "milliseconds" << std::endl;


    auto start_naive = std::chrono::steady_clock::now();
    for (auto ball_ptr : balls) {
        intersections_naive.clear();
        for (auto other_ball_ptr : balls) {
            if (ball_ptr != other_ball_ptr) {
                if (Ball::intersect(ball_ptr, other_ball_ptr)) {
                    intersections_naive.push_back(other_ball_ptr);
                }
            }
        }
    }
    auto finish_naive = std::chrono::steady_clock::now();
    std::cout << "Naive took " << std::chrono::duration_cast<std::chrono::milliseconds>(finish_BVH - start_BVH).count() << "milliseconds" << std::endl;
}

// checking BVH class [End] 

// checking graphics (openGL) [Begin]

void application_3(int argc, char ** argv) {
    int choice = atoi(argv[1]);

    if (choice == 1) {
        GLfloat points[] = {0.0f, 0.5f, 0.0f, 0.5f, -0.5f, 0.0f, -0.5f, -0.5f, 0.0f};
        
        Camera camera(std::array<GLfloat, 3> {0.0, 0.0, -1.0}, std::array<GLfloat, 3> {0.0, 0.0, 1.0}, std::array<GLfloat, 3> {0.0, 1.0, 0.0});
        Drawer drawer(camera);

        drawer.Init("VertexShader_simple.glsl", "FragmentShader_simple.glsl");

        GLint colour_location = glGetUniformLocation(drawer.opengl_program, "input_color");
        assert(colour_location > -1);

        glUseProgram(drawer.opengl_program);
        glUniform4f(colour_location, 1.0f, 0.0f, 0.0f, 1.0f);       
        
        while (!glfwWindowShouldClose(drawer.window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            GLfloat lines[] = {0.0f, 0.5f, 0.0f, 0.5f, -0.5f, 0.0f, -0.5f, -0.5f, 0.0};
            int n_points = 3;
            std::vector<std::array<std::array<GLfloat, 3>, 2>> lines_vector;
            
            for (int line_index = 0; line_index < n_points - 1; line_index++) {
                int first_point_index = line_index;
                int second_point_index = line_index + 1;

                std::array<GLfloat, 3> first_point;
                std::array<GLfloat, 3> second_point;

                for (int point_index : {0, 1, 2}) {
                    first_point[point_index] = lines[3 * first_point_index + point_index];
                    second_point[point_index] = lines[3 * second_point_index + point_index];
                }
                auto line = std::array<std::array<GLfloat, 3>, 2> {first_point, second_point};
                lines_vector.push_back(line);
            }

            drawer.drawer_graphics.show_wireframe(drawer.window, drawer.opengl_program, lines_vector);
            glfwPollEvents();
            if (GLFW_PRESS == glfwGetKey(drawer.window, GLFW_KEY_ESCAPE)) {
                glfwSetWindowShouldClose(drawer.window, 1);
            }
            glfwSwapBuffers(drawer.window);
        }

        drawer.clear();
    } else if (choice == 2) {
        
        Camera camera(std::array<GLfloat, 3> {0.0, 0.0, -1.0}, std::array<GLfloat, 3> {0.0, 0.0, 1.0}, std::array<GLfloat, 3> {0.0, 1.0, 0.0});
        Drawer drawer(camera);

        drawer.Init("VertexShader_simple.glsl", "FragmentShader_simple.glsl");

        GLint colour_location = glGetUniformLocation(drawer.opengl_program, "input_color");
        assert(colour_location > -1);

        glUseProgram(drawer.opengl_program);
        glUniform4f(colour_location, 1.0f, 0.0f, 0.0f, 1.0f);       
        
        while (!glfwWindowShouldClose(drawer.window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            GLfloat triangles[] = {0.0f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f};
            int n_points = 3;
            std::vector<Triangle> triangles_vector;
            std::vector<Triangle> normals_vector;
            
            for (int triangle_index = 0; triangle_index < n_points / 3; triangle_index++) {
                
                std::array<std::array<GLfloat, 3>, 3> positions;
                std::array<std::array<GLfloat, 3>, 3> normals;
                
                for (int point_count : {0, 1, 2}) {
                    for (int index_count : {0, 1, 2}) {
                        positions[point_count][index_count] = triangles[9 * triangle_index + 3 * point_count + index_count];
                        normals[point_count][index_count] = 0.0;
                    }
                }

                triangles_vector.push_back(Triangle(positions));
                normals_vector.push_back(Triangle(normals));
            }

            drawer.drawer_graphics.Show_triangles(drawer.window, drawer.opengl_program, triangles_vector, normals_vector);

            glfwPollEvents();
            if (GLFW_PRESS == glfwGetKey(drawer.window, GLFW_KEY_ESCAPE)) {
                glfwSetWindowShouldClose(drawer.window, 1);
            }
            glfwSwapBuffers(drawer.window);
        }

        drawer.clear();
    } else if (choice == 3) {
        std::array<GLfloat, 3> camera_origin {1.0, 1.0, 1.0};
        std::array<GLfloat, 3> looking_towards {0.0, 0.0, 0.0};

        Camera camera(camera_origin, MATH::multiply(1.0 / MATH::length(MATH::sub(looking_towards, camera_origin)), MATH::sub(looking_towards, camera_origin)), std::array<GLfloat, 3> {0.0, 1.0, 0.0});
        Drawer drawer(camera);

        drawer.Init("VertexShader_world.glsl", "FragmentShader_world.glsl");

        
        GLint colour_location = glGetUniformLocation(drawer.opengl_program, "input_color");
        assert(colour_location > -1);

        glUseProgram(drawer.opengl_program);
        glUniform4f(colour_location, 1.0f, 0.0f, 0.0f, 1.0f);

        GLint projection_matrix_location = glGetUniformLocation(drawer.opengl_program, "projection_matrix");
        assert(projection_matrix_location > -1);
        GLint camera_matrix_location = glGetUniformLocation(drawer.opengl_program, "camera_matrix");
        assert(camera_matrix_location > -1);

        glUniformMatrix4fv(projection_matrix_location, 1, GL_FALSE, camera.projection_matrix(0.01, 100.0, 60.0).data());
        glUniformMatrix4fv(camera_matrix_location, 1, GL_FALSE, camera.column_major().data());


        while (!glfwWindowShouldClose(drawer.window)) {

            static GLfloat previous_seconds = glfwGetTime();
            GLfloat current_time_seconds = glfwGetTime();
            GLfloat elapsed_time = current_time_seconds - previous_seconds;
            previous_seconds = current_time_seconds;

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            GLfloat triangles[] = {0.0f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f};
            int n_points = 3;
            std::vector<Triangle> triangles_vector;
            std::vector<Triangle> normals_vector;
            
            for (int triangle_index = 0; triangle_index < n_points / 3; triangle_index++) {
                
                std::array<std::array<GLfloat, 3>, 3> positions;
                std::array<std::array<GLfloat, 3>, 3> normals;
                
                for (int point_count : {0, 1, 2}) {
                    for (int index_count : {0, 1, 2}) {
                        positions[point_count][index_count] = triangles[9 * triangle_index + 3 * point_count + index_count];
                        normals[point_count][index_count] = 0.0;
                    }
                }

                triangles_vector.push_back(Triangle(positions));
                normals_vector.push_back(Triangle(normals));
            }

            drawer.drawer_graphics.Show_triangles(drawer.window, drawer.opengl_program, triangles_vector, normals_vector);

            glfwPollEvents();
            if (GLFW_PRESS == glfwGetKey(drawer.window, GLFW_KEY_ESCAPE)) {
                glfwSetWindowShouldClose(drawer.window, 1);
            }
            glfwSwapBuffers(drawer.window);
        }

        drawer.clear();
    } else if (choice == 4) {
        std::array<GLfloat, 3> camera_origin {1.0, 1.0, 1.0};
        std::array<GLfloat, 3> looking_towards {0.0, 0.0, 0.0};

        Camera camera(camera_origin, MATH::multiply(1.0 / MATH::length(MATH::sub(looking_towards, camera_origin)), MATH::sub(looking_towards, camera_origin)), std::array<GLfloat, 3> {0.0, 1.0, 0.0});
        Drawer drawer(camera);

        drawer.Init("VertexShader_world.glsl", "FragmentShader_world.glsl");

        
        GLint colour_location = glGetUniformLocation(drawer.opengl_program, "input_color");
        assert(colour_location > -1);

        glUseProgram(drawer.opengl_program);
        glUniform4f(colour_location, 1.0f, 0.0f, 0.0f, 1.0f);

        GLint projection_matrix_location = glGetUniformLocation(drawer.opengl_program, "projection_matrix");
        assert(projection_matrix_location > -1);
        GLint camera_matrix_location = glGetUniformLocation(drawer.opengl_program, "camera_matrix");
        assert(camera_matrix_location > -1);

        glUniformMatrix4fv(projection_matrix_location, 1, GL_FALSE, camera.projection_matrix(0.01, 100.0, 60.0).data());
        glUniformMatrix4fv(camera_matrix_location, 1, GL_FALSE, camera.column_major().data());


        while (!glfwWindowShouldClose(drawer.window)) {

            static GLfloat previous_seconds = glfwGetTime();
            GLfloat current_time_seconds = glfwGetTime();
            GLfloat elapsed_time = current_time_seconds - previous_seconds;
            previous_seconds = current_time_seconds;

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            GLfloat triangles[] = {0.0f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f};
            int n_points = 3;
            std::vector<Triangle> triangles_vector;
            std::vector<Triangle> normals_vector;
            
            for (int triangle_index = 0; triangle_index < n_points / 3; triangle_index++) {
                
                std::array<std::array<GLfloat, 3>, 3> positions;
                std::array<std::array<GLfloat, 3>, 3> normals;
                
                for (int point_count : {0, 1, 2}) {
                    for (int index_count : {0, 1, 2}) {
                        positions[point_count][index_count] = triangles[9 * triangle_index + 3 * point_count + index_count];
                        if (index_count != 2) {
                            normals[point_count][index_count] = 0.0;
                        } else {
                            normals[point_count][index_count] = 1.0;
                        }
                    }
                }

                triangles_vector.push_back(Triangle(positions));
                normals_vector.push_back(Triangle(normals));
            }

            drawer.drawer_graphics.Show_triangles(drawer.window, drawer.opengl_program, triangles_vector, normals_vector);

            glfwPollEvents();
            if (GLFW_PRESS == glfwGetKey(drawer.window, GLFW_KEY_ESCAPE)) {
                glfwSetWindowShouldClose(drawer.window, 1);
            }
            glfwSwapBuffers(drawer.window);
        }

        drawer.clear();
    }
}

// checking graphics/mesh (openGL) [End]

// for testing purposes [End]




// main application [Begin]
// NOT IMPLEMENTED!
// All the classes are done. I have to combine them and implement camera rotation.
// main applicaiton [End]



int main(int argc, char ** argv) {

/**
    application_1(argc, argv);
**/
/**
    to provide argv = [N_points, Sedd, bounding_box_edge_length]
    application_2(argc, argv);
**/
/**
    to provide argv = [choise in 1, 2, 3]
    application_3(argc, argv);
**/    
    return 0;
}
