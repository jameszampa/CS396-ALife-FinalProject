from dm_control import mjcf
from genotype import SpherePart


def translate_genotype_to_phenotype_recursive(model, parent_body, sphere_part, direction, motor_strength_dict, strength_dict):
    if sphere_part.created:
        return
    sphere_part.created = True
    child_body = parent_body.add('body', name=f"{sphere_part.name}", pos=sphere_part.pos)
    child_body.add('geom', type='sphere', size=(sphere_part.size, sphere_part.size, sphere_part.size))
    opposite_dir = (-direction[0], -direction[1], -direction[2])
    joint = child_body.add('joint', type=sphere_part.joint_type, axis=opposite_dir)
    if sphere_part.frozen_edges[direction]:
        # motor = model.actuator.add('motor', joint=joint, ctrllimited='true', ctrlrange=(0, 0))
        # motor_strength_dict[motor.name] = 0
        pass
    else:
        motor = model.actuator.add('motor', joint=joint, ctrllimited='true', ctrlrange=(-100, 100), name=f"{sphere_part.name}_motor")
        # print(sphere_part.edge_strength[direction])
        # print(motor.name)
        motor_strength_dict[motor.name] = strength_dict[direction]
        # print(sphere_part.edge_strength[direction])
    for next_direction, next_child in sphere_part.edges.items():
        if next_child is not None:
            translate_genotype_to_phenotype_recursive(model, child_body, next_child, next_direction, motor_strength_dict, next_child.edge_strength)


def translate_genotype_to_phenotype(genotype: SpherePart, plane_height=1, strength_dict=None):
    motor_strength_dict = {}

    model = mjcf.RootElement()
    plane = model.worldbody.add('geom', type='plane', size=(100, 100, 0.1), pos=(0, 0, plane_height - 1))

    root_body = model.worldbody.add('body', name='body0', pos=(0, 0, 0))
    root_body.add('joint', type='free')
    root_body.add('geom', type='sphere', size=(genotype.size, genotype.size, genotype.size))
    genotype.created = True

    for direction, child in genotype.edges.items():
        if child is not None:
            translate_genotype_to_phenotype_recursive(model, root_body, child, direction, motor_strength_dict, strength_dict)

    return model, motor_strength_dict


def calculate_plane_height(genotype: SpherePart):
    return calculate_plane_height_recursive([genotype], genotype, genotype.lowest_z)


def calculate_plane_height_recursive(all_sphere_parts, sphere_part, lowest_z):
    for direction, next_sphere_part in sphere_part.edges.items():
        if next_sphere_part is not None and next_sphere_part not in all_sphere_parts:
            all_sphere_parts.append(next_sphere_part)
            if next_sphere_part.lowest_z < lowest_z:
                lowest_z = next_sphere_part.lowest_z
            lowest_z = calculate_plane_height_recursive(all_sphere_parts, next_sphere_part, lowest_z)
    return lowest_z
