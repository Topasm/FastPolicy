# ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# can also choose libero_spatial, libero_object, etc.
task_suite_name = "libero_10"
task_id = 0
resize = 256


benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path(
    "bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " +
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": resize,
    "camera_widths": resize
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
# for benchmarking purpose, we fix the a set of initial states
init_states = task_suite.get_task_init_states(task_id)
env.set_init_state(init_states[0])

dummy_action = [0.] * 7
replay_images = []

if task_suite_name == "libero_spatial":
    max_steps = 220  # longest training demo has 193 steps
elif task_suite_name == "libero_object":
    max_steps = 280  # longest training demo has 254 steps
elif task_suite_name == "libero_goal":
    max_steps = 300  # longest training demo has 270 steps
elif task_suite_name == "libero_10":
    max_steps = 520  # longest training demo has 505 steps
elif task_suite_name == "libero_90":
    max_steps = 400  # longest training demo has 373 steps

for step in range(max_steps):
    obs, reward, done, info = env.step(dummy_action)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    replay_images.append(img)
env.close()
