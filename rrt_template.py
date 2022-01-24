import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, \
    set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
from utils import draw_sphere_marker

########## RRT motion Planning for PR2 arm by Yang-Lun Lai ###############
np.random.seed(17)
joint_names = (
'l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_elbow_flex_joint', 'l_upper_arm_roll_joint', 'l_forearm_roll_joint',
'l_wrist_flex_joint')

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2table.json')

    # define active DoFs
    joint_names = ('l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_elbow_flex_joint', 'l_upper_arm_roll_joint',
                   'l_forearm_roll_joint', 'l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i]: (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit,
                                     get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in
                    range(len(joint_idx))}

    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, 1.19, -1.548, 1.557, -1.32, -0.1928)))
    print ("start running RRT...")
    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []

    def limit_check(config, joint_limits):
        for i in range(len(joint_limits)):
            if (config[i] < joint_limits[joint_names[i]][0] or config[i] > joint_limits[joint_names[i]][1]) and i != 4:
                return False
        return True

    def distance(now, next):
        distance_temp = 0
        weights = [0.08, 2.0, 0.55, 0.35, 0.25, 0.08]
        for i in range(6):
            distance_temp = distance_temp + weights[i] * abs(next[i] - now[i])
        return distance_temp

    def get_rand(joint_limits, goal_config, goal_bias):
        prob = np.random.random(1)  # To make probability of picking the goal node instead of the random one.
        if prob <= goal_bias:
            return goal_config
        else:
            q_random = [0, 0, 0, 0, 0, 0]
            for i in range(len(joint_limits)):
                temp_low_lim, temp_up_lim = joint_limits[joint_names[i]]
                q_random[i] = round(random.uniform(temp_low_lim, temp_up_lim), 2)
            return (q_random[0], q_random[1], q_random[2], q_random[3], q_random[4], q_random[5])

    def get_near(explored_nodes, q_random):
        d_list = []
        for node in explored_nodes:
            d_list.append(distance(node, q_random))
        min_ind = np.argmin(d_list)
        return explored_nodes[min_ind]

    def get_new(near, random, step_size):
        d = distance(near, random)
        if d < step_size:
            return random
        else:
            new_config = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                new_config[i] = near[i] + round((random[i] - near[i]) * (step_size / (2 * d)), 2)
            return (new_config[0], new_config[1], new_config[2], new_config[3], new_config[4], new_config[5])

    step_size = 0.05  # (rad)
    goal_bias = 0.1  # 10%
    tolerance = step_size
    node_id = 0
    root = -1
    explored = []
    explored.append(start_config)
    parent = {}  # a dictionary key: tuple(a config), value: tuple(parent's config)
    parent[start_config] = root
    collision_times = 0
    final_config = (-1, -1, -1, -1, -1, -1)  # initialization
    finished = 0

    # RRT-Connect
    while finished == 0:
        random_config = get_rand(joint_limits, goal_config, goal_bias)
        near_config = get_near(explored, random_config)

        if collision_times > 200:  # in order not to stuck in a weired place
            explored = [start_config]
            collision_times = 0

        connect_times = 0
        while connect_times < 200:  # to prevent the random sample is too far away

            # check if the near node hit obstacle or out of limit
            if collision_fn(near_config) == True or limit_check(near_config, joint_limits) == False:
                collision_times = collision_times + 1
                break  # get random node again

            if distance(near_config, random_config) < step_size:  # reach the random sample
                new_config = random_config
                parent[new_config] = near_config
                explored.append(new_config)
                break  # get random node again

            else:  # have not reach the random sample, then extend!
                new_config = get_new(near_config, random_config, step_size)

            if collision_fn(new_config) == True or limit_check(new_config, joint_limits) == False:  # hit an obstacle
                collision_times = collision_times + 1
                break  # get random node again

            parent[new_config] = near_config
            explored.append(new_config)
            near_config = new_config  # the new node then become the nearest node to the random node.

            connect_times = connect_times + 1

            if distance(near_config, goal_config) < step_size:  # then, we check whether we arrive goal now
                current_config = near_config
                final_config = near_config
                while parent[current_config] != root:
                    path.insert(0, current_config)
                    current_config = parent[current_config]
                path.append(goal_config)
                finished = 1
                break

    PR2 = robots['pr2']
    set_joint_positions(PR2, joint_idx, goal_config)
    ee_pose = get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))
    radius = 0.03
    color = (0, 0, 1, 1) # draw the goal
    draw_sphere_marker(ee_pose[0], radius, color)
    set_joint_positions(PR2, joint_idx, start_config)
    wait_if_gui()
    print("Start drawing the position of the left end-effector in black")
    radius = 0.02
    color = (0, 0, 0, 1)
    PR2 = robots['pr2']
    for pose in path:
        set_joint_positions(PR2, joint_idx, pose)
        ee_pose = get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))
        draw_sphere_marker(ee_pose[0], radius, color)
    wait_if_gui()
    # Path Smoothing for 150 iterations
    iteration = 150
    origin_path = path
    new_path = origin_path
    for i in range(iteration):
        sub_path = []
        path_len = len(new_path)
        num1, num2 = random.sample(range(0, path_len), 2)  # pick two points on the path randomly
        if num1 == num2:
            num2 = random.sample(range(num1 + 1, path_len), 1)
        qone = min(num1, num2)
        qtwo = max(num1, num2)
        current = new_path[qone]
        subgoal = new_path[qtwo]
        connected = 0
        while connected == 0:
            if collision_fn(current) == True or limit_check(current, joint_limits) == False:
                break

            if distance(current, subgoal) < step_size:
                del new_path[qone + 1:qtwo]  # eliminate the old path between q1 and q2
                parent[subgoal] = current
                connected = 1

                sub_current_config = subgoal
                sub_current_path = []

                while parent[sub_current_config] != parent[new_path[qone]]:
                    sub_current_path.insert(0, sub_current_config)
                    sub_current_config = parent[sub_current_config]

                for i in range(len(sub_current_path)):
                    new_path.insert(qone + i + 1, sub_current_path[i])

            new_config = get_new(current, subgoal, step_size)

            if collision_fn(new_config) == True or limit_check(new_config, joint_limits) == False:
                break
            parent[new_config] = current
            current = new_config

    set_joint_positions(PR2, joint_idx, start_config)
    wait_if_gui()
    PR2 = robots['pr2']
    print("Start drawing the shortcut-smoothed path of the end-effector in red")
    radius = 0.02
    color = (1, 0, 0, 1)
    for pose in new_path:
        set_joint_positions(PR2, joint_idx, pose)
        ee_pose = get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))
        draw_sphere_marker(ee_pose[0], radius, color)

    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], joint_idx, new_path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
