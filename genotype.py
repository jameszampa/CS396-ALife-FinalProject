import random
import numpy as np


class SpherePart:
    def __init__(self, name, size, pos, absolute_pos):
        self.name = name
        self.size = size
        self.pos = pos
        self.joint_type = "hinge"
        self.created = False
        self.parent_direction = None
        self.parent = None
        self.fitness = 0.0
        self.absolute_pos = absolute_pos
        self.lowest_z = absolute_pos[2] + self.pos[2] - self.size

        # Key is the direction and Value is a SpherePart
        self.edges = {}
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for direction in directions:
            self.edges[direction] = None

        self.frozen_edges = {}
        for direction in directions:
            self.frozen_edges[direction] = False

        self.edge_strength = {}
        for direction in directions:
            self.edge_strength[direction] = np.random.uniform(-1.0, 1.0) * 10


    def add_edge(self, direction, sphere_part):
        self.edges[direction] = sphere_part
        opposite_direction = (-direction[0], -direction[1], -direction[2])
        sphere_part.edges[opposite_direction] = self
        sphere_part.parent_direction = opposite_direction
        sphere_part.parent = self
        
    
    def remove_edge(self, direction, parent):
        # Check if the edge exists before trying to remove it
        if direction in self.edges and self.edges[direction] == parent:
            # Save the parent node before deleting the edge
            parent_node = self.edges[direction]
            del self.edges[direction]
            opposite_direction = (-direction[0], -direction[1], -direction[2])
            # Check if the edge exists in the parent node before trying to remove it
            if opposite_direction in parent_node.edges and parent_node.edges[opposite_direction] == self:
                del parent_node.edges[opposite_direction]


def get_all_sphere_parts(all_sphere_parts, sphere_part):
    for direction, next_sphere_part in sphere_part.edges.items():
        if next_sphere_part is not None and next_sphere_part not in all_sphere_parts:
            all_sphere_parts.append(next_sphere_part)
            get_all_sphere_parts(all_sphere_parts, next_sphere_part)
    return all_sphere_parts


def add_node_mutation(head, part, node_name):
    # If we do not have any None values then we cannot add a new node
    num_none = 0
    for edge in part.edges.values():
        if edge is None:
            num_none += 1
    if num_none == 0:
        return part
    
    # Select a random direction
    potential_directions = [key for key, value in part.edges.items() if value is None]
    direction = random.choice(potential_directions)
    # Select a random size
    random_size = random.uniform(0.5, 1.0)
    position = np.array(direction) * (random_size + part.size)
    absolute_pos = part.absolute_pos + position
    if check_if_sphere_part_collides_with_other_sphere_parts(head, absolute_pos, random_size):
        return part
    # Create a new SpherePart
    new_sphere_part = SpherePart(node_name, random_size, position, absolute_pos)
    part.add_edge(direction, new_sphere_part)
    return part


def check_if_sphere_part_collides_with_other_sphere_parts(head, absolute_pos, size):
    for part in get_all_sphere_parts([head], head):
        distance = np.linalg.norm(absolute_pos - part.absolute_pos)
        if distance < (size + part.size):
            return True
    return False


def flip_freeze_edge_mutation(part):
    # Select a random direction
    potential_directions = [key for key, value in part.edges.items()]
    direction = random.choice(potential_directions)
    # Freeze the edge
    part.frozen_edges[direction] = not part.frozen_edges[direction]
    return part


def remove_node_mutation(random_part):
    random_part.remove_edge(random_part.parent_direction, random_part.parent)
    return random_part


def flip_joint_type_mutation(part):
    if part.joint_type == "hinge":
        part.joint_type = "ball"
    else:
        part.joint_type = "hinge"
    return part


def adjust_motor_strength_mutation(part):
    # Select a random direction
    potential_directions = [key for key, value in part.edges.items() if value is not None]
    if len(potential_directions) == 0:
        return part
    direction = random.choice(potential_directions)
    # Adjust the motor strength
    part.edge_strength[direction] = random.uniform(-1.0, 1.0) * 10
    # print(part.edge_strength[direction])
    return part


def mutate(head, part, all_sphere_parts):
    mutation_functions = [add_node_mutation, flip_freeze_edge_mutation, remove_node_mutation, flip_joint_type_mutation, adjust_motor_strength_mutation]
    mutation_function = random.choice(mutation_functions)
    if mutation_function == add_node_mutation:
        max_name_num = 0
        for part in all_sphere_parts:
            if part.name.startswith("body"):
                part_num = int(part.name[4:])
                if part_num > max_name_num:
                    max_name_num = part_num
        part_num = max_name_num
        return mutation_function(head, part, f"body{part_num + 1}")
    return mutation_function(part)


def get_creature_length(genotype):
    return get_creature_length_recursive([genotype], genotype)


def get_creature_length_recursive(all_sphere_parts, sphere_part):
    for direction, next_sphere_part in sphere_part.edges.items():
        if next_sphere_part is not None and next_sphere_part not in all_sphere_parts:
            all_sphere_parts.append(next_sphere_part)
            get_creature_length_recursive(all_sphere_parts, next_sphere_part)
    return len(all_sphere_parts)