# **POWERplay:** A toolchain to study AI power-seeking

![POWERplay banner image](img/powerplay-banner-image.png)

**POWERplay**  is an open-source toolchain that makes it easy to study power-seeking behavior in reinforcement learning agents. POWERplay was developed by [Gladstone AI](https://www.gladstone.ai/), an AI safety company. It's primarily intended as a research tool.

## FAQs:

### Why did you build this?

[There's been a](https://www.amazon.com/Superintelligence-Dangers-Strategies-Nick-Bostrom/dp/1501227742) [long-standing](https://www.vanityfair.com/news/2017/03/elon-musk-billion-dollar-crusade-to-stop-ai-space-x) [debate](https://blogs.scientificamerican.com/observations/dont-fear-the-terminator/) [in AI](https://www.alignmentforum.org/posts/WxW6Gc6f2z3mzmqKs/debate-on-instrumental-convergence-between-lecun-russell) over whether highly capable AI systems will, or won't, be dangerous to human beings by default. As frontier AI capabilities [continue to accelerate](https://www.aitracker.org/), the question becomes an increasingly important one.

This debate has largely centered on the concern that future AI systems may try to accumulate power for themselves, even when we don't want them to. Some people believe that if an AI system is capable enough, it will try to accumulate power almost regardless of what its end-goal is. This belief is sometimes called the [_instrumental convergence thesis_](https://nickbostrom.com/superintelligentwill.pdf). Here's a [well-written example](https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-2.html) of it.

So far, evidence for the instrumental convergence thesis [has been](https://www.aaai.org/ocs/index.php/WS/AAAIW16/paper/view/12634/12347) [mainly](https://arxiv.org/pdf/1912.01683.pdf) [theoretical](https://arxiv.org/pdf/2206.11831.pdf). To the best of our knowledge, POWERplay is among the first attempts to directly investigate the thesis experimentally.

### What can I do with it?

The main thing you do with POWERplay is calculate how much **instrumental value** a state holds for an RL agent. "Instrumental value" is just a fancy term for "power" or "potential to achieve a wide variety of goals." For example, if you have a billion dollars, you're in a state that has high instrumental value. You can achieve a lot of different goals with a billion dollars. But if you're dead, you're in a state that has low instrumental value. Dead people, unfortunately, can't accomplish much.

Below is an example of POWERplay in action. An agent moves through a maze, and POWERplay calculates how much instrumental value the agent would have at each position in the maze:

![7x7 maze example](img/7x7-maze-example.png)

You can see that the agent has high instrumental value at junctions, because it can quickly reach many goals from a junction. But it has low instrumental value at dead ends, because it can't quickly reach as many goals from a dead end.

You can also use POWERplay to study complicated interactions between multiple RL agents. In fact, this is probably its most interesting use case. For example, you can use POWERplay to investigate what situations drive agents to compete or collaborate with each other and why.

### What have you done with it?

We've used POWERplay to investigate a few topics already:

- [How instrumental convergence works in single-agent settings](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems)
- [How agents often compete for power even if their goals are unrelated](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems)
- [How physical interactions between agents affect power-seeking](https://www.alignmentforum.org/posts/nisaAr7wMDiMLc2so/instrumental-convergence-scale-and-physical-interactions)

In case you're wondering, the results we've seen so far seem to tentatively support the instrumental convergence thesis. But more research is needed before we can say anything conclusive. In fact, accelerating that research is the main reason we decided to open-source POWERplay.

### Where can I learn more?

If you'd like to better understand the theory behind POWERplay, you can check out the definitions POWERplay uses to calculate instrumental value. [Here's the definition it uses](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems#2__Single_agent_POWER) in single-agent settings, and [here's the one it uses](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2__Multi_agent_POWER__human_AI_scenario) in multi-agent setings. And if you really like math, [here's an appendix](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#Appendix_A__Detailed_definitions_of_multi_agent_POWER) with all the juicy details.

## Installation, setup, and testing

👉 _These installation instructions have been tested with Python 3.8.9 on MacOS. If you have a different system, you may need to change some of these steps._

1. Clone this repo and `cd` into the repo directory:
    ```
    % git clone https://github.com/gladstoneai/POWERplay.git
    ```

    ```
    % cd POWERplay
    ```

2. Ensure Homebrew and Graphviz are installed to enable MDP visualization. Run:
  
    ```
    % /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

    ```
    % brew install graphviz
    ```

    **Note:** When you attempt to install graphviz, you may encounter the following error:
    
    ```
    Error: graphviz: Invalid bottle tag symbol
    ```

    If this happens, you may be able to fix it by running the following command:

    ```
    % brew update-reset
    ```

    and _then_ installing graphviz afterwards:

    ```
    % brew install graphviz
    ```

3. Activate `virtualenv`. Run:

    ```
    % python3 -m venv venv
    % source venv/bin/activate
    ```

4. Ensure pip is up to date and download required packages. Run:

    ```
    % pip install --upgrade pip
    % pip install -r requirements.txt
    ```

5. Create an account on [Weights & Biases](https://wandb.ai/site) if you don't have one already. Then create a file in the main directory of this repo called `settings.json` with the following format:

    ```
    {
        "public": {
          "WANDB_DEFAULT_ENTITY": $YOUR_WANDB_USERNAME
        },
        "private": {
            "WANDB_API_KEY": $YOUR_WANDB_KEY
        }
    }
    ```

    You can find `$YOUR_WANDB_USERNAME` in your Weights & Biases Settings. Log into Weights & Biases and [go to your settings page](https://wandb.ai/settings). Your username will be listed under "USERNAME" in the "Profile" section of your settings.

    You can also find `$YOUR_WANDB_KEY` in your Weights & Biases [Settings page](https://wandb.ai/settings). Scroll down to "Danger Zone" in the Settings page. Under "API keys", either copy your existing key if you have one, or click "New key" and copy it.

    When you paste your username and API key into the `settings.json` file above, **make sure to put them in double quotes**. For example, if your username is `bob-bobson`, and your API key is `abc123`, your finished `settings.json` file would look like:

    ```
    {
        "public": {
          "WANDB_DEFAULT_ENTITY": "bob-bobson"
        },
        "private": {
            "WANDB_API_KEY": "abc123"
        }
    }
    ```

6. Run the `test_vanilla()` function that will calculate the POWER for each state in the [MDP](https://en.wikipedia.org/wiki/Markov_decision_process) below, plot the results, and post them to your Weights & Biases account.

    ![Example MDP from Optimal Policies Tend to Seek POWER, https://arxiv.org/abs/1912.01683](img/opttsp-mdp-example.png)

    To run `test_vanilla()`, do the following:

    ```
    % python3 -i main.py
    ```

    Then:

    ```
    >>> base.test_vanilla()
    ```

7. Confirm that the output you get is consistent. You should see something like:

    ```
    wandb: Appending key for api.wandb.ai to your netrc file: /Users/bobbobson/.netrc
    wandb: Starting wandb agent 🕵️
    2022-10-18 10:24:26,174 - wandb.wandb_agent - INFO - Running runs: []
    2022-10-18 10:24:26,688 - wandb.wandb_agent - INFO - Agent received command: run
    2022-10-18 10:24:26,688 - wandb.wandb_agent - INFO - Agent starting run with config:
        convergence_threshold: 0.0001
        discount_rate: [0.1, '0p1']
        mdp_graph: mdp_from_paper
        num_reward_samples: 10000
        num_workers: 10
        random_seed: 0
        reward_distribution: {'allow_all_equal_rewards': True, 'default_dist': {'dist_name': 'uniform', 'params': [0, 1]}, 'state_dists': {'ℓ_◁': {'dist_name': 'uniform', 'params': [-2, 0]}, '∅': {'dist_name': 'uniform', 'params': [-1, 1]}}}
    2022-10-18 10:24:26,690 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env python sweep.py --convergence_threshold=0.0001 "--discount_rate=[0.1, '0p1']" --mdp_graph=mdp_from_paper --num_reward_samples=10000 --num_workers=10 --random_seed=0 "--reward_distribution={'allow_all_equal_rewards': True, 'default_dist': {'dist_name': 'uniform', 'params': [0, 1]}, 'state_dists': {'ℓ_◁': {'dist_name': 'uniform', 'params': [-2, 0]}, '∅': {'dist_name': 'uniform', 'params': [-1, 1]}}}"
    wandb: Currently logged in as: bob-bobson (use `wandb login --relogin` to force relogin)
    wandb: wandb version 0.13.4 is available!  To upgrade, please run:
    wandb:  $ pip install wandb --upgrade
    wandb: Tracking run with wandb version 0.12.7
    wandb: Syncing run splendid-sweep-1
    wandb:  View project at https://wandb.ai/bob-bobson/uncategorized
    wandb:  View sweep at https://wandb.ai/bob-bobson/uncategorized/sweeps/7w9xi059
    wandb:  View run at https://wandb.ai/bob-bobson/uncategorized/runs/mey3t37m
    wandb: Run data is saved locally in /Users/bobbobson/your-folder/POWERplay/wandb/run-20221018_102428-mey3t37m
    wandb: Run `wandb offline` to turn off syncing.


    Computing POWER samples:

    2022-10-18 10:24:31,697 - wandb.wandb_agent - INFO - Running runs: ['mey3t37m']
    Running samples 10000 / 10000

    Run complete.

    Rendering plots...

    [...etc...]

    wandb: Waiting for W&B process to finish, PID 47164... (success).
    wandb:                                                                                
    wandb: Synced 5 W&B file(s), 13 media file(s), 0 artifact file(s) and 0 other file(s)
    wandb: Synced denim-sweep-3: https://wandb.ai/bob-bobson/uncategorized/runs/s1zj29jd
    wandb: Find logs at: ./wandb/run-20221018_102523-s1zj29jd/logs/debug.log
    wandb: 
    2022-10-18 10:25:48,684 - wandb.wandb_agent - INFO - Cleaning up finished run: s1zj29jd
    2022-10-18 10:25:49,567 - wandb.wandb_agent - INFO - Agent received command: exit
    2022-10-18 10:25:49,567 - wandb.wandb_agent - INFO - Received exit command. Killing runs and quitting.
    wandb: Terminating and syncing runs. Press ctrl-c to kill.
    ```

    Navigating to your Sweeps view (using the link above under "View sweep at") should show the following:

    ![power-example-sweep-wandb](img/power_example_sweep_wandb.png)

    This sweep iterates over three discount rate values: 0.1, 0.3, and 0.5. The config YAML file for the test sweep is located at `configs/test/test_vanilla.yaml`.

8. Repeat steps 5 and 6 for the four other available test functions:

    - `base.test_gridworld()`, which tests the ability to run and visualize gridworlds
    - `base.test_stochastic()`, which tests the simulation loop with stochastic MDPs
    - `base.test_multiagent()`, which tests the simulation loop in multi-agent settings
    - `base.test_reward_correlation()`, which tests the ability to simulate multi-agent settings with agents [whose reward functions are partially correlated](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#3_1_Multi_agent_reward_function_distributions)

    You can also run all these test functions in succession with the following command:

    ```
    >>> base.run_all_tests()
    ```

    The results from all these runs, including plots, will appear in the `/expts` folder of this repo's root directory.

## Quickstart

## Basic usage

### Running an experiment

🟣 To run an experiment, use the `launch.launch_sweep()` function. This function takes a config filename as its only required argument. The config file is a YAML file that contains the parameters for the experiment.

For example:

  ```
  >>> launch.launch_sweep('test_vanilla.yaml')
  ```

The YAML file is the canonical description of the sweep for your experiment, and YAML files corresponding to individual runs of your sweep are saved in the `expts` and `wandb` directories.

🔵 Here are the input arguments to `launch_sweep()` and what they mean (the YAML API is described afterwards):

(Listed as `name [type] (default): description`.)

- `sweep_config_filename [str, required]`: The name of the YAML file that contains the configuration for your sweep. This file should be located in the `configs` directory, but you don't need to include the `configs/` prefix in the filename.

  Typical value: `'test_vanilla.yaml'`

- `sweep_local_id [str] (time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))`: A unique identifier for your sweep. This is used to name the directory in which the sweep and its runs are saved. This id will also show up in the names of your runs in the W&B UI.

  Typical value: `20211123111452`

- `entity [str] (data.get_settings_value(data.WANDB_ENTITY_PATH, settings_filename=data.SETTINGS_FILENAME))`: The W&B entity that is running the sweep. This is used to properly save your sweep in W&B.

  Typical value: `'bob-bobson'`

- `project [str] ('uncategorized')`: The W&B project that is running the sweep. This is used to properly save your sweep in W&B. If a project of that name is not owned by the same `entity` that is running the sweep, a new project will be created.

  Typical value: `'uncategorized'`

- `sweep_config_folder [str] (data.SWEEP_CONFIGS_FOLDER)`: The folder in which the sweep config file is located. This is used to find the config file.

  Typical value: `'configs'`

- `output_folder_local [str] (data.EXPERIMENT_FOLDER)`: The folder in which the experiment and its runs will be saved.

  Typical value: `'expts'`

- `plot_as_gridworld [bool] (False)`: Whether to visualize the POWER plots on a gridworld. Note that if you aren't running a gridworld MDP, setting this to `True` will crash the run.

  Typical value: `False`

- `plot_correlations [bool] (False)`: Whether to visualize the correlation plots. Sometimes you have so many states that all the correlation plots don't render in reasonable time during the sweep (because if there are `N` states, there are `N^2` correlation plots).

  Typical value: `False`

- `announce_when_done [bool] (False)`: Exactly what it sounds like.

  Typical value: `False`

- `environ [environ] (os.environ)`: The complete shell environment of the run; includes the environment variables that will be used to run the experiment. In general you shouldn't need to change this.

  Typical value: `environ({ 'USER': 'bob_bobson', ... })`

You can find examples of sweep configuration files in the `configs` folder. The file `test_vanilla.yaml` defines a sweep for the single-agent case, with a single Agent H MDP graph given by the parameter `mdp_graph` (see below). The file `test_run_multi_actual.yaml` defines a single run for the multi-agent case, with an Agent H MDP graph given by `mdp_graph_agent_H`, and Agent A MDP graph given by `mdp_graph_agent_A`, and an Agent A fixed policy graph given by `policy_graph_agent_A`.

Here are the entries of the sweep YAML file:

- `name [str]`: The name of your sweep, typically separated by underscores.

  Typical value: `'test_sweep'`

- `description [str]`: A description of your sweep.

  Typical value: `'This is a test sweep.'`

- `program`: The Python script that runs your sweep. This should always be set to `'sweep.py'`.

  Typical value: `'sweep.py'`

- `parameters`: A series of entries and values that define the sweep. Includes fixed values and values that vary over the sweep. Below are the parameters and their syntax **for fixed values**:

  - (Single-agent only) `mdp_graph`: The name of the file in the `mdps` folder that corresponds to the MDP graph you want to use. This name should not include the `.gv` extension. **Do not include this parameter if you are running a multi-agent sweep.**

    Typical value: `'mdp_from_paper'`
  
  - (Multi-agent only) `mdp_graph_agent_H`: The name of the file in the `mdps` folder that corresponds to the MDP graph you want to use for Agent H. This name should not include the `.gv` extension. **Do not include this parameter if you are running a single-agent sweep.**

    Typical value: `'stoch_gridworld_1x3_agent_H'`
  
  - (Multi-agent only) `mdp_graph_agent_A`: The name of the file in the `mdps` folder that corresponds to the MDP graph you want to use for Agent A. This name should not include the `.gv` extension. **Do not include this parameter if you are running a single-agent sweep.**
  
    Typical value: `'stoch_gridworld_1x3_agent_A'`
  
  - (Multi-agent only) `policy_graph_agent_A`: The name of the file in the `policies` folder that corresponds to the policy graph you want to use for Agent A. This name should not include the `.gv` extension. **Do not include this parameter if you are running a single-agent sweep.**
    
    Typical value: `'joint_1x3_gridworld_agent_A_move_left'`
  
  - `discount_rate`: The discount rate for the MDP.

    Typical value: `0.1`
  
  - `reward_distribution`: A list of distributions for the reward of each state. We have one `default_dist`, which defines the default reward distribution and is assumed to be iid over all states, and we also have a list of `state_dists`, which define state-specific reward distributions for all states that do not have the default distribution. Each distribution is defined by a `dist_name`, which is the name of that base distribution as found in `DISTRIBUTION_DICT` in `utils/dist.py`; and a `params` list, which is a list of parameters for that distribution.

    Typical value:

    ```
    default_dist:
        dist_name:
          "uniform"
        params:
          [0, 1]
    state_dists:
      "∅":
        dist_name:
          "uniform"
        params:
          [-1, 1]
      "ℓ_◁":
        dist_name:
          "uniform"
        params:
          [-2, 0]
    allow_all_equal_rewards:
      True
    ```
  
  - `num_reward_samples`: The number of samples to draw from the reward distribution for each state in the POWER calculations.

    Typical value: `10000`
  
  - `convergence_threshold`: The convergence threshold for the value iteration algorithm.

    Typical value: `0.0001`
  
  - `num_workers`: The number of workers to use in multiprocessing.

    Typical value: `10`
  
  - `random_seed`: The random seed to use for the experiment. If `null`, then the random seed is not fixed. Normally set to a number, to make it easier to reproduce the experiment later.

    Typical value: `0`

  Note that the above parameter values are set with different syntax depending on whether the value is **fixed**, or whether it **varies** over the course of the sweep. If the value is **fixed**, then the syntax puts the actual value of the parameter _under a `value` key_ in the parameter dictionary, like so:

  ```
  discount_rate:
    value:
      0.1
  ```

  If the value **varies**, then the parameter takes on multiple values over the course of a sweep. The way to specify this is to create a `values` key under which you list each value of the parameter, _and also a `names` key_ under which you list the names of the corresponding values in order to allow each run of your sweep to be named according to the parameter values it corresponds to. For example:

  ```
  discount_rate:
    values:
      - 0.1
      - 0.2
      - 0.3
    names:
      - "0p1"
      - "0p2"
      - "0p3"
  ```

  (Note that it's best to use "p" instead of "." for decimals, since the parameter names are going to be used in filenames.)

🟢 The `launch.launch_sweep()` function returns no output. Instead, it saves the results of the sweep to the `output_folder_local` folder, and to the `/wandb` folder. The rendered correlation plots associated with each run of the sweep are saved in a subfolder that corresponds to their run.

### Saving and loading MDP graphs

🟣 To save a new MDP graph for later experiments, use `data.save_graph_to_dot_file()` to save a NetworkX graph as a `dot` file in a target folder. For example, the following code creates and saves the MDP graph from Figure 1 of _Optimal policies tend to seek power_:

```
>>> import networkx as nx
>>> new_mdp = nx.DiGraph([
        ('★', '∅'), ('★', 'ℓ_◁'), ('★', 'r_▷'),
        ('∅', '∅'),
        ('ℓ_◁', 'ℓ_↖'), ('ℓ_◁', 'ℓ_↙'),
        ('ℓ_↙', 'ℓ_↖'), ('ℓ_↙', 'ℓ_↙'),
        ('r_▷', 'r_↗'), ('r_▷', 'r_↘'),
        ('r_↘', 'r_↘'), ('r_↘', 'r_↗'),
        ('r_↗', 'r_↗'), ('r_↗', 'r_↘'),
        ('ℓ_↖', 'ℓ_↙'))
    ], name='POWER paper MDP')
>>> data.save_graph_to_dot_file(new_mdp, 'mdp_from_paper')
```

🔵 Here are the input arguments to `data.save_graph_to_dot_file()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.DiGraph, required]`: The NetworkX [DiGraph](https://networkx.org/documentation/stable/reference/classes/digraph.html) you want to save as your MDP. This should be a directed graph, with nodes representing states and edges representing transitions. Every state must have at least one outgoing edge, even if the state points to itself (i.e., the state has a self-loop).

  Typical value: `mdp.quick_graph_to_mdp(nx.petersen_graph(), name='Petersen graph')`

- `mdp_filename [str, required]`: The name of the file to save the MDP graph as. This should be a filename without an extension, and should be unique among all MDP graphs you have saved. (If the name you save with is the same as the name of an existing file, this will overwrite the existing file without warning.)

  Typical value: `'petersen_graph'`

- `folder [str] (data.MDPS_FOLDER)`: The folder to save the graph in. Note that you can save policy graphs in the same way as MDP graphs, and if so, you should generally save these in the `policies` folder.

  Typical value: `'mdps'`

🟢 The `data.save_graph_to_dot_file()` function returns no output. Instead, it saves the MDP graph to the `data.MDPS_FOLDER` folder for future use.

🟣 To load a previously saved NetworkX graph, use `data.load_graph_from_dot_file()`. For example, the following code loads the MDP graph that's used in the `test.test_vanilla()` integration test:

```
>>> new_mdp = data.load_graph_from_dot_file('mdp_from_paper')
```

🔵 Here are the input arguments to `data.load_graph_from_dot_file()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_name [str, required]`: The file name of the graph you want to load. This **should not** include the `.gv` extension.

  Typical value: `'mdp_from_paper'`

- `folder [str] (data.MDPS_FOLDER)`: The folder to load the graph from. Note that you can load policy graphs with this function too; if you're doing that, you should load them from the `policies` folder. 

  Typical value: `'mdps'`

🟢 Here is the output to `data.load_graph_from_dot_file()`:

- `output_graph [networkx.DiGraph]`: The loaded NetworkX graph. This may be either an MDP or a policy graph.

### Creating a simple MDP graph

🟣 You can create any kind of simple MDP graph manually by directly calling the NetworkX `DiGraph()` API. For example, the following code creates the MDP graph from Figure 1 of _Optimal policies tend to seek power_:

```
>>> import networkx as nx
>>> new_mdp = nx.DiGraph([
        ('★', '∅'), ('★', 'ℓ_◁'), ('★', 'r_▷'),
        ('∅', '∅'),
        ('ℓ_◁', 'ℓ_↖'), ('ℓ_◁', 'ℓ_↙'),
        ('ℓ_↙', 'ℓ_↖'), ('ℓ_↙', 'ℓ_↙'),
        ('r_▷', 'r_↗'), ('r_▷', 'r_↘'),
        ('r_↘', 'r_↘'), ('r_↘', 'r_↗'),
        ('r_↗', 'r_↗'), ('r_↗', 'r_↘'),
        ('ℓ_↖', 'ℓ_↙'))
    ], name='POWER paper MDP')
```

For full documentation on the NetworkX `DiGraph()` API, see [here](https://networkx.org/documentation/stable/reference/classes/digraph.html).

🟣 For quick tests, you can use one of the [prepackaged NetworkX graph topologies](https://networkx.org/documentation/stable/tutorial.html?highlight=petersen_graph#graph-generators-and-graph-operations) (such as the Petersen graph), and convert these to a compatible directed graph using `mdp.quick_graph_to_mdp()`:

  ```
  >>> import networkx as nx
  >>> petersen_mdp = mdp.quick_graph_to_mdp(nx.petersen_graph(), name='Petersen graph')
  ```

🔵 Here are the input arguments to `mdp.quick_graph_to_mdp()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.Graph OR networkx.DiGraph, required]`: The NetworkX graph you want to convert. This can be either a `Graph` or a `DiGraph`. In either case, it will be converted to a `DiGraph`. You'll often want to use some of the pre-packaged NetworkX graphs as input for quick testing.

  Typical value: `nx.petersen_graph()`

- `name [str] ('')`: The name you want to give the output graph.

  Typical value: `'Petersen graph'`

🟢 Here is the output to `mdp.quick_graph_to_mdp()`:

- `output_graph [networkx.DiGraph]`: The output NetworkX `DiGraph`. Note that using `quick_graph_to_mdp()` on an undirected graph makes the resulting output graph not just _directed_, but also _acyclic_. It also makes sure that every non-terminal node has at least one outbound edge or self-loop.

### Creating a gridworld MDP graph

🟣 To create a gridworld MDP, use `mdp.construct_gridworld()`. For example, the following code creates the gridworld pictured below:

```
>>> gridworld = mdp.construct_gridworld(10, 10, squares_to_delete=[['(0, 0)', '(3, 2)'], ['(6, 4)', '(8, 7)']])
```

![gridworld-example](img/gridworld_example.png)

🔵 Here are the input arguments to `mdp.construct_gridworld()` and what they mean:

(Listed as `name [type] (default): description`.)

- `num_rows [int, required]`: The maximum number of rows in your gridworld.

  Typical value: `5`

- `num_cols [int, required]`: The maximum number of columns in your gridworld.

  Typical value: `5`

- `name [str] ('custom gridworld')`: The name you want to give your gridworld.

  Typical value: `'3x3 gridworld'`

- `squares_to_delete [list] ([])`: A list of 2-tuples, where each 2-tuple is a pair of coordinates (in **string** format) for the edges of a square you want to delete from your gridworld. For example, if you want to delete the square with the top-left corner at (0, 0) and the bottom-right corner at (2, 2), then you would use `squares_to_delete=[['(0, 0)', '(2, 2)']]`. This format allows us to quickly construct gridworlds with interesting structures.

  Typical value: `[['(0, 0)', '(3, 2)'], ['(6, 4)', '(8, 7)']]`

🟢 Here is the output to `mdp.construct_gridworld()`:

- `gridworld_graph [networkx.DiGraph]`: An MDP graph representing the gridworld you created. Each square has a self-loop, and connections to the squares above, beneath, to the left, and to the right of it (if they exist).

  The states of the gridworld MDP are strings indicating the coordinates of each cell in the gridworld. For example, the state `'(0, 0)'` represents the cell at the top-left corner of the gridworld.

### Creating a stochastic MDP graph

🟣 To create a stochastic MDP, start from an existing MDP graph and use `mdp.mdp_to_stochastic_graph()`. For example, the following code changes the simple `gridworld_selfloop_3x3` graph to a stochastic format:

``` 
>>> stochastic_mdp = mdp.mdp_to_stochastic_graph(data.load_graph_from_dot_file('gridworld_selfloop_3x3'))
```

🔵 Here are the input arguments to `mdp.mdp_to_stochastic_graph()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.DiGraph, required]`: The NetworkX [DiGraph](https://networkx.org/documentation/stable/reference/classes/digraph.html) you want to convert into stochastic format. It should have a graph structure like this:

  ![simple-mdp-example](img/simple_mdp_example.png)
  
  Here the ovals represent states and the arrows represent transitions. The states are labeled with their names; the MDP above is a representation of the 3x3 gridworld. Note that all of the states have self-loops.

  Typical value: `data.load_graph_from_dot_file('gridworld_selfloop_3x3')`

🟢 Here is the output to `mdp.mdp_to_stochastic_graph()`:

- `stochastic_graph [networkx.DiGraph]`: A NetworkX graph representing the new MDP, in stochastic format. This format explicitly represents the transition probabilities from one state to another in the graph. Here is an example:

  ![stochastic-mdp-gridworld](img/stochastic_mdp_gridworld.png)

  Here the squares represent states, the circles represent actions, and the arrows represent transitions. The numbers on each arrow represent the probability of that action-state transition.

🟣 To convert a **gridworld** specifically to a stochastic graph, you'll typically want to use `mdp.gridworld_to_stochastic_graph()`. This is a more powerful function than `mdp.mdp_to_stochastic_graph()`, because it includes options for adding stochastic noise to some of the gridworld state transitions — but it's limited to taking gridworlds as inputs, rather than general MDPs. For example, the following code takes the simple `gridworld_selfloop_3x3` gridworld MDP, and converts it into a gridworld with left-facing stochastic "wind" in all cells:

``` 
>>> stochastic_gw_with_left_wind = mdp.gridworld_to_stochastic_graph(data.load_graph_from_dot_file('gridworld_selfloop_3x3'), stochastic_noise_level=0.2, noise_bias={ 'left': 0.2 })
```

🔵 Here are the input arguments to `mdp.gridworld_to_stochastic_graph()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.DiGraph, required]`: The NetworkX gridworld [DiGraph](https://networkx.org/documentation/stable/reference/classes/digraph.html) you want to convert into stochastic format.

  Typical value: `data.load_graph_from_dot_file('gridworld_selfloop_3x3')`

🟢 Here is the output to `mdp.mdp_to_stochastic_graph()`:

- `stochastic_graph [networkx.DiGraph]`: A NetworkX graph representing the new gridworld MDP, in stochastic format. This format explicitly represents the transition probabilities from one state to another in the graph. Here is an example:

  ![stochastic-mdp-gridworld](img/stochastic_mdp_gridworld.png)

  Here the squares represent states, the circles represent actions, and the arrows represent transitions. The numbers on each arrow represent the probability of that action-state transition.

🟣 To add a state (and its downstream actions) to a stochastic MDP, start from an existing stochastic MDP, and use use `mdp.add_state_action()`. For example, the following code adds a state with actions `'L'`, `'H'`, and `'R'` to the MDP graph:

```
>>> stochastic_mdp = mdp.add_state_action(nx.DiGraph(), '2', {
        'L': { '1': 1 },
        'H': { '1': 0.5, '2': 0.5 },
        'R': { '2': 0.2, '3': 0.8 }
    })
```

🔵 Here are the input arguments to `mdp.add_state_action()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.DiGraph, required]`: The NetworkX [DiGraph](https://networkx.org/documentation/stable/reference/classes/digraph.html) you want to add the state to. This should be an MDP in stochastic format, i.e., formatted as in the picture below:

  ![stochastic-mdp-example](img/stochastic_mdp_example.png)

  Typical value: `nx.DiGraph()`

- `state_to_add [str, required]`: The state whose outgoing transitions you want to add to the MDP graph.

  Typical value: `'1'`

- `action_dict [dict, required]`: A dictionary of actions and their corresponding transitions. The keys of this dictionary are actions, and the values are dictionaries of states and their corresponding probabilities.

  Typical value:
  ```
    {
        'L': { '1': 1 },
        'H': { '1': 0.5, '2': 0.5 },
        'R': { '2': 0.2, '3': 0.8 }
    }
  ```

- `check_closure [bool] (False)`: Whether or not to throw a warning if the MDP graph is not closed. "Not closed" means that the MDP incldues states that have no outgoing transitions. MDPs that aren't closed can't be used for experiments and will cause an error in POWER sweeps.

  Typical value: `False`

🟢 Here is the output to `mdp.add_state_action()`:

- `new_mdp_graph [networkx.DiGraph]`: An MDP graph representing the new MDP, with the new state, actions, and outgoing transitions added to it.

### Creating a multiagent MDP graph

🟣 You can create a two-agent MDP graph from the perspective of either agent by starting from a "naive" single-agent MDP graph and applying `multi.create_multiagent_graph()` to it. This function takes the state set of the input MDP graph, and takes the outer product of those states by assuming that both Agent H and Agent A are moving on the same state set. So if the original graph was a 1x2 gridworld with cells `'(0, 0)'` and `'(0, 1)'`, the output graph would have states `'(0_H, 0_A)'`, `'(0_H, 1_A)'`, `'(1_H, 0_A)'`, and `'(1_H, 1_A)'`. For example, the following code creates a multiagent MDP graph from the perspective of Agent H, out of a 3x3 gridworld graph in stochastic format:

``` 
>>> mdp_graph_H = multi.create_multiagent_graph(mdp.gridworld_to_stochastic_graph(data.load_graph_from_dot_file('gridworld_selfloop_3x3')), acting_agent_is_H=True)
```

🔵 Here are the input arguments to `mdp.create_multiagent_graph()` and what they mean:

(Listed as `name [type] (default): description`.)

- `single_agent_graph [networkx.DiGraph, required]`: The single-agent MDP graph that you want to convert into multiagent format. **This graph should already be in stochastic format,** so if it isn't, you should apply either `mdp.mdp_to_stochastic_graph()` or `mdp.gridworld_to_stochastic_graph()` to it first (depending on whether the original is a gridworld or not).

  Typical value: `mdp.gridworld_to_stochastic_graph(data.load_graph_from_dot_file('gridworld_selfloop_3x3'))`

- `acting_agent_is_H [bool] (True)`: Set this to `True` if you want the output multiagent graph to be from the perspective of Agent H, and `False` if you want it from the perspective of Agent A. The agent you pick will be the one whose actions will affect the states; so if you choose the perspective of Agent H, your output graph will have transitions like `'(0_H, 0_A)' == 'right' ==> '(1_H, 0_A)'`, but will _not_ have transitions like `'(0_H, 0_A)' == 'right' ==> '(0_H, 1_A)'`.

  Typical value: `True`

🟢 Here is the output to `mdp.create_multiagent_graph()`:

- `multiagent_graph [networkx.DiGraph]`: A NetworkX graph representing the multiagent MDP from the perspective of the agent you selected in `acting_agent_is_H`. This MDP is in stochastic format, and its states are the outer product of Agent H and Agent A positional states. Here is an example of a multiagent MDP from the perspective of Agent H:

  ![multiagent-mdp-H](img/joint_multiagent_mdp_example.png)

  These MDPs represent two agents on the same underlying state set: a 2x2 gridworld. Notice that the joint states of H and A are labelled as, e.g., `(0, 0)_H^(0, 0)_A`. This is the standard notation that's used under the hood to label joint states in a multiagent system.

### Creating a fixed policy graph

🟣 To run a multiagent experiment, you need to create a fixed policy for Agent A — otherwise the state transitions from the perspective of Agent H won't be fully defined. You can quickly create a random policy for Agent A using the `base.build_quick_random_policy()` function. (You have to apply this function to an MDP **over joint states** that's **from the perspective of the agent you want the policy to be for** — in practice, that means you apply this to the output of the `multi.create_multiagent_graph()` function.) For example, the following code creates a random Agent A policy over the 3x3 gridworld graph:

``` 
>>> policy_A = base.build_quick_random_policy(multi.create_multiagent_graph(mdp.gridworld_to_stochastic_graph(data.load_graph_from_dot_file('gridworld_selfloop_3x3')), acting_agent_is_H=False))
```

Notice that `acting_agent_is_H` is set to `False`, which is what defines this policy as being as belonging to Agent A.

🔵 Here are the input arguments to `base.build_quick_random_policy()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.DiGraph, required]`: The MDP graph over joint states, from the perspective of the agent whose policy you want as output. Normally this will be the output of a call to the `multi.create_multiagent_graph()` function.

  Typical value: `multi.create_multiagent_graph(mdp.gridworld_to_stochastic_graph(data.load_graph_from_dot_file('gridworld_selfloop_3x3')), acting_agent_is_H=False)`

🟢 Here is the output to `base.build_quick_random_policy()`:

- `policy_graph [networkx.DiGraph]`: A NetworkX graph representing the policy of the agent whose perspective the input MDP graph was from. The policy graph format is pretty similar to the MDP graph format. Here's an example, for an Agent A policy:

  ![policy-graph-A](img/policy_graph_example_A.png)

  As you'd expect, the rectangles represent joint states, the circles represent actions, and the arrows represent transitions. The numbers on each arrow represent the probability of taking a given action from a given state, under that policy. (Note that the output of `base.build_quick_random_policy()` will be a _random_ policy, with equal probabilities of each action from a given state, while the example shown above is deterministic.)

🟣 The `base.build_quick_random_policy()` function returns a random policy. But in general, we want to test all sorts of different policies in our multiagent runs. The way to do this is to start from a random policy, and run `policy.update_state_actions()` on it until you end up with the policy you want. For example, the following code updates the Agent A random policy on a 3x3 gridworld to deterministically move to the right when H and A are both at cell `(0, 0)`:

``` 
>>> policy_A_right = policy.update_state_actions(base.build_quick_random_policy(multi.create_multiagent_graph(mdp.gridworld_to_stochastic_graph(data.load_graph_from_dot_file('gridworld_selfloop_3x3')), acting_agent_is_H=False)), '(0 ,0)_H^(0, 0)_A', { 'right': 1, 'down': 0, 'stay': 0 })
```

🔵 Here are the input arguments to `policy.update_state_actions()` and what they mean:

(Listed as `name [type] (default): description`.)

- `policy_graph [networkx.DiGraph, required]`: The policy graph you want to update.

  Typical value: `base.build_quick_random_policy(multi.create_multiagent_graph(mdp.gridworld_to_stochastic_graph(data.load_graph_from_dot_file('gridworld_selfloop_3x3')), acting_agent_is_H=False))`

- `state [str, required]`: The state whose action probabilities you want to update in your new policy.

  Typical value: `'(0, 0)_H^(0, 0)_A'`

- `new_policy_actions [dict, required]`: A dictionary listing all the allowed actions from `state`, along with the probabilities you want to assign to them under the new policy. You need to list _all_ the allowed actions from `state`, and assign them _all_ new probabilities that, of course total to 1.

  Typical value: `{ 'right': 1, 'down': 0, 'stay': 0 }`

🟢 Here is the output to `policy.update_state_actions()`:

- `policy_graph [networkx.DiGraph]`: The NetworkX graph representing the new policy after the update. This will have exactly the same format (and the same state set) as the input policy graph, just with the action probabilities from the target state modified according to the `new_policy_actions` dict you assigned.

## Advanced wordflows

### Launching a real sweep

Here's an example of launching a real sweep, assuming you've got a project called `'power-project'` in your W&B profile. When you have lots of states in your MDP (>30 or so) it can be best to set `plot_correlations=False` because the correlation plots will otherwise take way too long to render.

```
launch.launch_sweep(
      'sweep-2x3_GRIDWORLD_MULTIAGENT_CLOCKWISE_MOVING_POLICY_GAMMA_SWEEP-distribution_uniform_0t1-samples_40k.yaml',
      entity=data.get_settings_value('public.WANDB_COLLAB_ENTITY'),
      project='power-project',
      plot_as_gridworld=False,
      plot_correlations=False,
      announce_when_done=True
  )
```

### Creating multiagent policies at scale

Multiagent policies are defined on the _joint_ MDP state space of Agent H and Agent A. That means if the base MDP has `N` states, the joint MDP will have `N^2` states. Because you have to manually define a multiagent policy on each state that can make it pretty painful to scale to big systems.

You can get around this problem by defining a single-agent policy first, _and only then_ using `multi.create_multiagent_graph()` on that policy to take the outer product of the state space the policy is defined on. This will work for any multiagent policy where the action sets of the two agents are orthogonal.

For example, here's how you'd define an Agent A policy on a 2x3 gridworld. Normally you'd have to define this policy on 36 states (H at (0, 0) + A at (0,0), H at (0,0) + A at (0, 1), etc.). But this shortcut lets us define the policy for A on just 6 states, then take the outer product to get the whole multiagent policy in one line.

```
gw = mdp.construct_gridworld(2, 3, name='2x3 gridworld')
st_gw = mdp.gridworld_to_stochastic_graph(gw)
pol = base.build_quick_random_policy(st_gw)

pol = policy.update_state_actions(pol, '(0, 0)', { 'stay': 0, 'right': 1, 'down': 0 })
pol = policy.update_state_actions(pol, '(0, 1)', { 'stay': 0, 'right': 1, 'down': 0, 'left': 0 })
pol = policy.update_state_actions(pol, '(0, 2)', { 'stay': 0, 'down': 1, 'left': 0 })
pol = policy.update_state_actions(pol, '(1, 0)', { 'stay': 0, 'up': 1, 'right': 0 })
pol = policy.update_state_actions(pol, '(1, 1)', { 'stay': 0, 'right': 0, 'up': 0, 'left': 1 })
pol = policy.update_state_actions(pol, '(1, 2)', { 'stay': 0, 'up': 0, 'left': 1 })

pol_A = multi.create_multiagent_graph(pol, acting_agent_is_H=False)
```

You can then create the multiagent MDPs as usual from the basic stochastic gridworld above:

```
mdp_H = multi.create_multiagent_graph(st_gw, acting_agent_is_H=True)
mdp_A = multi.create_multiagent_graph(st_gw, acting_agent_is_H=False)
```

### Plotting a policy sample

Sometimes, you'll want to plot a particular optimal policy that corresponds to a particular reward function sample. You can do this by chaining together `policy.sample_optimal_policy_data_from_run()` (which takes in a run identifier and the index of the reward sample you're interested in) and `viz.plot_policy_sample()` (which plots the policy sample along with its reward function at each state).

For example, here's how you might plot the optimal policy for the third reward sample of a run with `sweep_id` of `'20220525090545'` and `run_suffix` of `'discount_rate__0p1'`:

  ```
  sweep_id = '20220525090545'
  run_suffix = 'discount_rate__0p1'

  policy_graph, reward_function, discount_rate = policy.sample_optimal_policy_data_from_run(sweep_id, run_suffix)
  viz.plot_policy_sample(policy_graph, reward_function, discount_rate)
  ```

Note that if you're investigating a single-agent run, this will plot the optimal policy of the single agent; but if you're investigating a multiagent run, this will plot the optimal policy for **Agent H**. (The Agent A policy in multiagent is already given, so you can feed that into `viz.plot_mdp_or_policy()` directly.)
