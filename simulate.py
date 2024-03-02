# Written with assistance from Github Co-Pilot
import os
import copy
import time
import json
import random
import numpy as np
import mujoco.viewer
import dm_control.mujoco
from matplotlib import pyplot as plt
from genotype import SpherePart, mutate, get_all_sphere_parts, get_creature_length
from phenotype import translate_genotype_to_phenotype, calculate_plane_height


def simulate(model, motor_strength_dict):
    with open('creature.xml', 'w') as f:
        f.write(model.to_xml_string())
    
    m = dm_control.mujoco.MjModel.from_xml_path("creature.xml")
    d = dm_control.mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Set camera parameters
        # These parameters can be adjusted to change the camera angle and perspective
        viewer.cam.azimuth = 180  # Azimuthal angle (in degrees)
        viewer.cam.elevation = -20  # Elevation angle (in degrees)
        viewer.cam.distance = 15.0  # Distance from the camera to the target
        viewer.cam.lookat[0] = 0.0  # X-coordinate of the target position
        viewer.cam.lookat[1] = 0.0  # Y-coordinate of the target position
        viewer.cam.lookat[2] = 0.0  # Z-coordinate of the target position

        for i in range(20000):
            for m_name in motor_strength_dict:
                # print(m_name)
                i = dm_control.mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m_name)
                if motor_strength_dict[m_name] > 0:
                    d.ctrl[i] = 1000
                else:
                    d.ctrl[i] = -1000
            
            dm_control.mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(1/1000)

        viewer.close()


def simulate_no_viewer(genotype, model, motor_strength_dict):
    with open('creature.xml', 'w') as f:
        f.write(model.to_xml_string())
    
    m = dm_control.mujoco.MjModel.from_xml_path("creature.xml")
    d = dm_control.mujoco.MjData(m)

    fitness = 0

    # creature_velocity = 0
    step_idx = 0
    while (step_idx < 5000 or (d.qvel[0]) > 0.00) and step_idx < 20000:
        for m_name in motor_strength_dict:
            i = dm_control.mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m_name)
            if motor_strength_dict[m_name] > 0:
                d.ctrl[i] = 1000
            else:
                d.ctrl[i] = -1000

        dm_control.mujoco.mj_step(m, d)
        
        fitness += d.qvel[0] * 2
        fitness += d.qpos[0]
        step_idx += 1
    
    fitness /= step_idx
    fitness /= get_creature_length(genotype)
    # fitness += (d.qvel[0]) * 2
        
    print(f"Creature fitness: {fitness:.4f}")
    
    return fitness


def get_new_file_name():
    # creature_best_1.xml, creature_best_2.xml, etc.
    file_numbers = []
    for file_name in os.listdir("best_creatures"):
        if file_name.startswith("creature_best_") and file_name.endswith(".xml"):
            file_numbers.append(int(file_name[14:-4]))
    if len(file_numbers) == 0:
        return "best_creatures/creature_best_1.xml"
    return f"best_creatures/creature_best_{max(file_numbers) + 1}.xml"


def main():
    experiments = 1
    population_size = 100
    generations = 100
    percent_prev_gen_to_keep = 0.5
    mutation_rate = 10

    stages_of_evo = []

    for exp in range(experiments):
        print(f"Experiment {exp}")
        creatures = [SpherePart("body0", random.uniform(0.5, 1.0), (0, 0, 0), (0, 0, 0)) for _ in range(population_size)]
        
        for creature in creatures:
            for _ in range(random.randint(1, 5)):
                new_creature_parts = get_all_sphere_parts([creature], creature)
                mutate(creature, random.choice(new_creature_parts), new_creature_parts)
        
        best_fitness_history = []
        for gen_num in range(generations):
            for creature_idx, creature in enumerate(creatures):
                print(f"Generation {gen_num}, creature {creature_idx}")
                plane_height = calculate_plane_height(creature)
                model, motor_strength_dict = translate_genotype_to_phenotype(copy.deepcopy(creature), plane_height, creature.edge_strength)
                # print(motor_strength_dict)
                fitness = simulate_no_viewer(creature, model, motor_strength_dict)
                # Add fitness to creature
                creature.fitness = fitness
                creature.num_parts = get_creature_length(creature)
            
            # Sort creatures by fitness
            creatures.sort(key=lambda x: x.fitness, reverse=True)
            best_fitness_history.append(creatures[0].fitness)

            if (len(stages_of_evo) == 0) or (creatures[0].fitness != stages_of_evo[-1].fitness):
                stages_of_evo.append(creatures[0])
                creature = creatures[0]
                # print(f"Creature {i}: {creature.fitness:.4f}")
                plane_height = calculate_plane_height(creature)
                model, motor_strength_dict = translate_genotype_to_phenotype(copy.deepcopy(creature), plane_height, creature.edge_strength)
                # print(motor_strength_dict)
                file_name = get_new_file_name()
                file_name = file_name.replace("creature_best_", f"gen{gen_num}_creature_best_")
                with open(file_name, 'w') as f:
                    f.write(model.to_xml_string())
                with open(file_name[:-4] + ".json", 'w') as f:
                    f.write(json.dumps(motor_strength_dict, indent=4))
            # Keep the top % of creatures
            creatures = creatures[:int(population_size * percent_prev_gen_to_keep)]
            # Keep the best creature for each niche, being the num_parts of the creature
            # copy_creatures = []
            # for creature in creatures:
            #     if creature.num_parts not in [c.num_parts for c in copy_creatures]:
            #         copy_creatures.append(creature)
            #     elif creature.fitness > [c.fitness for c in copy_creatures if c.num_parts == creature.num_parts][0]:
            #         copy_creatures = [c for c in copy_creatures if c.num_parts != creature.num_parts]
            #         copy_creatures.append(creature)
            # creatures = copy_creatures
            # mean and std of fitness
            print(f"Mean fitness: {np.mean([creature.fitness for creature in creatures]):.4f}")
            print(f"Std fitness: {np.std([creature.fitness for creature in creatures]):.4f}")
            # Mutate the creatures
            new_creatures = []
            while len(new_creatures) + len(creatures) < population_size:
                new_creature = copy.deepcopy(random.choice(creatures))
                for _ in range(mutation_rate):
                    new_creature_parts = get_all_sphere_parts([new_creature], new_creature)
                    # print(new_creature.edge_strength)
                    mutate(new_creature, random.choice(new_creature_parts), new_creature_parts)
                    # print(new_creature.edge_strength)
                new_creatures.append(new_creature)
            creatures.extend(new_creatures)

        i = 0
        creature = creatures[i]
        print(f"Creature {i}: {creature.fitness:.4f}")
        plane_height = calculate_plane_height(creature)
        # print(f"Plane height: {plane_height}")

        model, motor_strength_dict = translate_genotype_to_phenotype(copy.deepcopy(creature), plane_height, creature.edge_strength)
        print(motor_strength_dict)
        file_name = get_new_file_name()
        with open(file_name, 'w') as f:
            f.write(model.to_xml_string())
        with open(file_name[:-4] + ".json", 'w') as f:
            f.write(json.dumps(motor_strength_dict, indent=4))
        # Create best_creature_fitness_history.png for each experiment
        plt.plot(best_fitness_history)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Best Creature Fitness History")
        plt.savefig(f"{file_name[:-4]}.png")
        # Save hyperparameters
        with open(file_name[:-4] + "_hyperparameters.json", 'w') as f:
            json.dump({
                "experiments": experiments,
                "population_size": population_size,
                "generations": generations,
                "percent_prev_gen_to_keep": percent_prev_gen_to_keep,
                "mutation_rate": mutation_rate
            }, f, indent=4)
        simulate(model, motor_strength_dict)

if __name__ == "__main__":
    main()