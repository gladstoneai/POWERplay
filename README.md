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

POWERplay can scale to handle MDPs with thousands of states. This scalability has enabled us to [discover emergent competitive interactions](https://www.alignmentforum.org/posts/nisaAr7wMDiMLc2so/instrumental-convergence-scale-and-physical-interactions#3_3_2_The_no_overlap_rule_increases_misalignment_between_far_sighted_agents) between RL agents that only seem to arise in fairly complicated settings.

In case you're wondering, the results we've seen so far seem to tentatively support the instrumental convergence thesis overall. But more research is needed before we can say anything conclusive. Accelerating this research is actually the main reason we've decided to open-source POWERplay.

### Where can I learn more?

If you'd like to better understand the theory behind POWERplay, you can check out the definitions POWERplay uses to calculate instrumental value. [Here's the definition it uses](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems#2__Single_agent_POWER) in the single-agent case, and [here's the one it uses](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2__Multi_agent_POWER__human_AI_scenario) in the multi-agent case. And if you really like math, [here's an appendix](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#Appendix_A__Detailed_definitions_of_multi_agent_POWER) with all the juicy details.

## Installation, setup, and testing

👉 _These installation instructions have been tested with Python 3.8.9 on MacOS. If you have a different system, you may need to change some of these steps._

1. Clone this repo and `cd` into the POWERplay directory:
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

To run your first sweep in POWERplay, first run the `main.py` file:

```
% python3 -i main.py
```

If you open the `configs/` folder in the POWERplay repo, you'll see a file there called `my_first_sweep.yaml`. This is the sweep we're going to run. In `main.py`, run the following command:

```
>>> base.launch_sweep('my_first_sweep.yaml', plot_as_gridworld=True)
```

This sweep will probably take a few minutes to run.

Once the sweep has finished, you'll be able to see the results, including pre-rendered figures and plots, in the `expts/` folder. The figures will appear inside a new folder called `{SWEEP_ID}-my_first_sweep`.

Look for files with the prefix `POWER_means`. These are plots of the POWERs (i.e., instrumental values) of each state for an agent on a simple maze gridworld. You'll notice these look different depending on the discount factor of the agent. To see an explanation of this difference, [check out this write-up](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems#3__Results).

## Testing and replication

### Testing

POWERplay includes five test functions in its core module `base`. You may have already run these as part of your [installation and setup](#installation-setup-and-testing). POWERplay also includes a wrapper function, `run_all_tests()`, that runs all these tests in sequence.

#### Basic test

🟣 If you want to test POWERplay's basic functionality, use `base.test_vanilla()`. Basic functionality means things like ingesting config files, multiprocessing to compute POWER values, generating output figures, connecting and uploading results to your [wandb](https://wandb.ai/site) account, and saving results and figures locally.

For example:

```
>>> base.test_vanilla()
```

This function executes the sweep defined in the config file `config/test/test_vanilla.yaml`. The outputs of the sweep, incluing figures, get saved locally under `expts/{SWEEP_ID}-test_vanilla`. (And get separately uploaded to wandb.)

#### Gridworld test

🟣 If you want to test POWERplay's ability to plot POWERs on gridworlds, use `base.test_gridworld()`. Instead of displaying POWERs in a bar plot, this function displays them as a heat map on a gridworld.

For example:

```
>>> base.test_gridworld()
```

This function executes the sweep defined in the config file `config/test/test_gridworld.yaml`. The agent's environment is a simple 3x3 gridworld.

The outputs of the sweep get uploaded to wandb and separately get saved locally under `expts/{SWEEP_ID}-test_gridworld`. Files with the prefix `POWER_means-agent_H` are figures that display POWERs as gridworld heatmaps.

#### Stochastic test

🟣 If you want to test POWERplay's ability to compute POWERs on _stochastic_ MDPs, use `base.test_stochastic()`.

For example:

```
>>> base.test_stochastic()
```

This function executes the sweep defined in the config file `config/test/test_stochastic.yaml`. The agent's environment is a 4-state MDP with stochastic transitions.

The outputs of the sweep get uploaded to wandb and separately get saved locally under `expts/{SWEEP_ID}-test_stochastic`.

#### Fixed-policy multi-agent test

🟣 If you want to test POWERplay's ability to compute POWERs on multi-agent MDPs, use `base.test_multiagent()`. Note that this computes Agent H POWERs assuming a _fixed policy_ for Agent A. So this test isn't "truly" multi-agent, because we can just _simulate_ Agent A's presence by incorporating its dynamics into Agent H's perceived transition function.

For example:

```
>>> base.test_multiagent()
```

This function executes two sweeps. The first, defined in the config file `config/test/test_multiagent_simulated.yaml`, _simulates_ the presence of Agent A, but does not execute any multi-agent code. The second sweep, defined in the config file `config/test/test_multiagent_actual.yaml`, runs Agent A using a defined fixed policy and multi-agent code. If POWERplay is functioning correctly, the outputs of these two sweeps should be exactly identical.

The outputs of these sweep get uploaded to wandb and separately get saved locally under `expts/{SWEEP_ID}-test_multiagent_simulated` and `expts/{SWEEP_ID}-test_multiagent_actual`.

#### Reward-correlation multi-agent test

🟣 To test POWERplay's ability to compute POWERs on multi-agent MDPs [where each agent may have a different reward function](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#3_1_Multi_agent_reward_function_distributions), use `base.test_reward_correlation()`. This test function uses a [novel definition of POWER](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2__Multi_agent_POWER__human_AI_scenario) (i.e., [instrumental value](#what-can-i-do-with-it)) that applies to certain important kinds of multi-agent systems.

For example:

```
>>> base.test_reward_correlation()
```

This function executes the sweep defined in the config file `config/test/test_reward_correlation.yaml`. The agents' environment is a 3x3 gridworld with joint states over the positions of the two agents — so 81 states in total, since either agent can occupy any of 9 positions on the 3x3 grid.

The outputs of the sweep get uploaded to wandb and separately get saved locally under `expts/{SWEEP_ID}-test_reward_correlation`. Files with the prefix `POWER_means-agent_H` show the POWERs of Agent H, and files with the prefix `POWER_means-agent_A` show the POWERs of Agent A. This test may take a few minutes to run.

#### Run all tests

🟣 You can run all of the above tests in succession using:

```
>>> base.run_all_tests()
```

### Replicating figures

🟣 You can use POWERplay to easily reproduce all the figures from our [first sequence](https://www.alignmentforum.org/s/HBMLmW9WsgsdZWg4R) of published experiments. We've included a function called `base.reproduce_figure()` which lets you input which figure you'd like to reproduce, and automatically runs a script to generate that figure for you.

For example:

```
>>> base.reproduce_figure(1, 2)
```

will reproduce Figures 2, 3, and 4 from [Part 1](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems) of our experiment series. Those three figures are all based on data from the same experiment sweep, so `reproduce_figure()` runs one sweep and reproduces all of these figures together to save time.

This function runs sweeps defined by the config files located in `configs/replication`.

Once it's finished rendering your figure, `reproduce_figure()` will print a message telling you the filename under which it's saved the figure. For example:

```
Figure available in temp/POWER_means-PART_1-FIGURE_1.png
```

This function saves figures in the `temp/` folder of the POWERplay repo.

🔵 Here are the inputs to `base.reproduce_figure()`.

(Listed as `name [type] (default): description`.)

- `post_number [int, required]`: A number that corresponds to the blog post whose figure you want to reproduce. [This post](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems) corresponds to `1`, [this post](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems) to `2`, and [this post](https://www.alignmentforum.org/posts/nisaAr7wMDiMLc2so/instrumental-convergence-scale-and-physical-interactions) to `3`.

  Typical value: `1`

- `fig_number [int, required]`: The number of the figure you'd like to reproduce in the post labeled by `post_number`. If multiple figures use the data from the same experiment sweep, a single run of `reproduce_figure()` will reproduce all those figures. For example, Figs 2, 3, and 4 in Part 1 all use the same sweep data. So if you run `reproduce_figure(1, 2)`, you'll generate all three figures.

  Note that `reproduce_figure(3, 1)` isn't supported, since Fig 1 in Part 3 is a copy of Fig 2 in Part 1. Keep in mind that some of the sweeps triggered by `reproduce_figure()` may take a few hours to run, especially those for `post_number=3`.

  Typical value: `1`

## Basic usage

To execute a complete experiment in POWERplay, you need to follow three steps:

1. Set up the environment for the experiment
2. Launch and run the experiment
3. Visualize the results of the experiment

Each of these steps is supported in the `base` module. We'll look at each one in turn.

### Setting up the environment

POWERplay supports both single-agent and multi-agent environments. In its multi-agent mode, POWERplay simulates two agents: a human agent (**"Agent H"**) and an AI agent (**"Agent A"**). POWERplay assumes that Agent A is dominant in this multi-agent setting, in the sense that Agent A learns much faster than Agent H. This assumption makes it possible to run experiments that are [relevant to long-term AI safety](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems#1__Introduction), while still remaining computationally tractable. See [this write-up](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2__Multi_agent_POWER__human_AI_scenario) for a full description of our multi-agent setting, and [this appendix](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#Appendix_A__Detailed_definitions_of_multi_agent_POWER) for full mathematical details.

In POWERplay, **setting up the environment** for an experiment means creating, at most, two objects:

1. **An MDP graph:** This is a graph that describes the [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP) your agent(s) operate in. The MDP graph defines the physics of the world your agent(s) live in. When an agent takes an action, your MDP graph describes how the world around it reacts to its action.

    MDP graphs get saved in the `mdps/` folder. When saving a multi-agent gridworld, best practice is to prefix the filename with `'joint_'`. For example, `'3x3_gridworld'` in the single-agent case, and `'joint_3x3_gridworld'` in the multi-agent case.

2. **A policy graph:** This is a graph that describes the [policy](https://stackoverflow.com/questions/46260775/what-is-a-policy-in-reinforcement-learning) that **Agent A** either a) follows at all times, or b) starts off with. We never have to define a policy graph for Agent H, because POWERplay runs its experiments from Agent H's "perspective" — meaning Agent H always _learns_ its policies during the experiment, rather than having them predefined. When we run a single-agent experiment, we're only running Agent H, so we don't need to define a policy graph.

    Policy graphs get saved in the `policies/` folder. When saving a policy graph, we **highly recommend** prefixing it with the name of the MDP it's associated with. For example, a uniform random policy acting on the `'joint_3x3_gridworld'` MDP might have the name `'joint_3x3_gridworld_agent_A_uniform_random'`.

#### Construct a single-agent gridworld MDP

🟣 To create a single-agent gridworld MDP, use `base.construct_single_agent_gridworld_mdp()`. You can quickly construct simple gridworlds with this function. For example, here's how to create and visualize a basic 3x4 gridworld:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4)
>>> base.view_gridworld(gridworld)
```

![3x4 gridworld example](img/3x4-gridworld-example.png)

You can also create more interesting gridworlds by "cutting out" squares from a bigger gridworld. For example, here's how to create (and visualize) a tiny maze from the 3x4 gridworld above:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> base.view_gridworld(gridworld)
```

![3x4 gridworld maze example](img/3x4-gridworld-maze-example.png)

We'll look at how to visualize gridworlds in more detail below, but for now we'll quickly note that you can also visualize this gridworld in its "raw" format as an MDP:

```
>>> base.plot_mdp_or_policy(gridworld)
```

![3x4 gridworld maze MDP example](img/3x4-gridworld-maze-example-mdp.png)

This is the same gridworld MDP as above, but with the state transitions and probabilities explicitly mapped out. This visualization contains _all_ the information POWERplay knows about an MDP, so it's the best one to use for deep troubleshooting.

The last thing you can do with `base.construct_single_agent_gridworld_mdp()` is inject **stochastic noise** into a gridworld MDP. Normally, when an agent takes an action on a gridworld, it moves deterministically in the direction that's consistent with its action. You can see this in the MDP graph above: if the agent starts from the `'(2, 0)'` state (bottom left of the grid) and takes the `'right'` action, it ends up in the `'(2, 1)'` state with probability 1.

But you can add noise to the MDP that makes this outcome non-deterministic. With noise, when the agent takes an action, it's no longer guaranteed to end up in the state that corresponds to the action it took. The more noise you add, the less influence the agent's action has over its next state. For example, here's how adding a moderate amount of noise changes the dynamics of our tiny maze gridworld above:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']], stochastic_noise_level=0.5)
>>> base.plot_mdp_or_policy(gridworld)
```

![3x4 gridworld maze MDP example with noise](img/3x4-gridworld-maze-noise-example-mdp.png)

Notice that the transitions look much more complicated than before. A `stochastic_noise_level` of 0.5 means that 0.5 units of probability are "spread out" equally over all the allowed next-states of our MDP. So this time, when our agent starts from the `'(2, 0)'` state and takes the `'right'` action, it ends up in the `'(2, 1)'` state with only probability 0.75, but ends up in the `'(2, 0)'` state with probability 0.25. (Those are the two states it's allowed to access from `'(2, 0)'`, and the stochastic noise level of 0.5 is divided equally between them.)

Finally, you can **bias** your gridworld MDP's stochastic noise in a particular direction. For example:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']], stochastic_noise_level=0.5, noise_bias={ 'left': 0.2 })
>>> base.plot_mdp_or_policy(gridworld)
```

![3x4 gridworld maze MDP example with biased noise](img/3x4-gridworld-maze-biased-noise-example-mdp.png)

The best way to build up an intuition for all these options is to try them out, and then visualize the resulting MDPs with `base.plot_mdp_or_policy()`.

🔵 Here are the input arguments to `base.construct_single_agent_gridworld_mdp()` and what they mean:

(Listed as `name [type] (default): description`.)

- `num_rows [int, required]`: The maximum number of rows in your gridworld.

  Typical value: `4`

- `num_cols [int, required]`: The maximum number of columns in your gridworld.

  Typical value: `4`

- `squares_to_delete [list] ([])`: A list of 2-tuples, where each 2-tuple is a pair of coordinates (in **string** format) that represent the edges of a square you want to delete from your gridworld. For example, if you want to delete the square with the top-left corner at (0, 0) and the bottom-right corner at (2, 2), then you would use `squares_to_delete=[['(0, 0)', '(2, 2)']]`. This format allows us to quickly construct gridworlds with interesting structures.

  Typical value: `[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']]`

- `stochastic_noise_level [float] (0)`: The level of stochastic noise you want to inject into your MDP graph. Setting this to `0` (the default) means no noise: when the agent takes an action, it will always end up in the state consistent with that action. So if an agent on a gridworld takes the `'right'` action, it will always move one cell to the right.

    Setting this to `1` means the agent's movement is fully determined by noise: the agent has no control at all over its movements, and its next state is completely random. So if an agent is at the upper left-hand corner of a gridworld — meaning it's allowed to take the `'down'`, `'right'` or `'stay'` actions — whatever action it takes, it will have a 1/3 probability each of moving down, moving right, or staying where it is.

    Setting stochastic noise to an intermediate value between `0` and `1` allocates that amount of probability to be divided equally between allowed actions, and allocates the rest to the action the agent actually takes. For example, suppose an agent is at the upper left-hand corner of a gridworld and `stochastic_noise_level=0.3`. Then if the agent takes the `'down'` action, it will have probability `0.3` allocated equally between the `'down'`, `'right'` and `'stay'` actions (`0.3 / 3 = 0.1` each), and the rest, `1 - 0.3 = 0.7`, allocated to the `'down'` action. So the final probabilities will be `'down'` with `0.7 + 0.1 = 0.8`, `'right'` with `0.1`, and `'stay'` with `0.1`.

    Typical value: `0.3`

- `noise_bias [dict] ({})`: A dict that defines the the direction and magnitude of the bias in the stochastic noise for your MDP. If empty, (i.e., `{}`) then `stochastic_noise_level` divides its probability equally between allowed next states. For example, suppose an agent is at the upper left-hand corner of a gridworld, and we have `stochastic_noise_level=0.3` with `noise_bias={}`. Then if the agent takes the `'down'` action, it will have probability `0.3` allocated equally between the `'down'`, `'right'` and `'stay'` actions (`0.3 / 3 = 0.1` each), and the rest, `1 - 0.3 = 0.7`, allocated to the `'down'` action. So the final probabilities will be `'down'` with `0.7 + 0.1 = 0.8`, `'right'` with `0.1`, and `'stay'` with `0.1`.

    If `noise_bias` includes a direction and magnitude, then the probability mass that corresponds with that magnitude gets allocated along the indicated direction. For example, if `stochastic_noise_level=0.3` and `noise_bias={ 'right': 0.3 }`, then the entire amount of stochastic noise gets allocated to the `'right'` action, rather than being divided equally among all legal actions. So if an agent is at the upper left-hand corner of a gridworld and takes the `'down'` action, it will have probability `0.3` allocated to the `'right'` action, and the rest, `1 - 0.3 = 0.7`, allocated to the `'down'` action. So the final probabilities will be `'down'` with `0.7`, and `'right'` with `0.3`.

    Typical value: `{ 'right': 0.3 }`

- `description [str] ('single-agent gridworld')`: A string that describes your gridworld.

    Typical value: `'3x3 single-agent gridworld with stochastic noise of 0.3'`

🟢 Here is the output to `base.construct_single_agent_gridworld_mdp()`:

- `mdp_graph [networkx.DiGraph]`: An MDP graph representing the gridworld you created.

  The states of single-agent gridworld MDP are strings that indicate where the agent is in the gridworld. For example, the state `'(0, 0)'` means that the agent is located at the top-left corner of the gridworld.

  The actions of a single-agent gridworld MDP are strings that indicate the allowed actions the agent can take from each state. Possible allowed actions are `'left'`, `'right'`, `'up'`, `'down'`, and `'stay'`. If a state is at the edge or corner of a gridworld, some of these actions will be forbidden.

  You can save your `mdp_graph` with `base.save_mdp_graph()`, view it as a gridworld with `base.view_gridworld()`, and view its full MDP with `base.plot_mdp_or_policy()`.

#### Construct a multi-agent gridworld MDP

🟣 To create a multi-agent gridworld MDP, use `base.construct_multiagent_gridworld_mdp()`. You can quickly construct simple multi-agent gridworlds with this function. For example, here's how to create and visualize a basic 3x4 multi-agent gridworld:

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4)
>>> base.view_gridworld(multiagent_gridworld)
```

![3x4 multi-agent gridworld example](img/3x4-multiagent-gridworld-example.png)

Notice that while the single-agent version of this gridworld has 3 x 4 = 12 states, the basic multi-agent version has 12 x 12 = 144 states. This is because each agent (Agent H and Agent A) can occupy any cell in the gridworld, including overlapping cells. In the visualization above, the red square represents the position of Agent A in each block.

You can also use `base.construct_multiagent_gridworld_mdp()` to create multi-agent gridworlds that have interesting shapes. You do this by "cutting out" squares of cells from the grid, using the same API as in the [single-agent case](#construct-a-single-agent-gridworld-mdp). For example, here's how to make a tiny multi-agent maze:

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> base.view_gridworld(multiagent_gridworld)
```

![3x4 multi-agent gridworld maze example](img/3x4-multiagent-gridworld-maze-example.png)

You can always visualize any multi-agent MDP in full, with all its transitions. For example:

```
>>> base.plot_mdp_or_policy(multiagent_gridworld)
```

![3x4 multi-agent gridworld maze MDP example](img/3x4-multiagent-gridworld-maze-example-mdp.png)

(You may need to open the image above separately to view it in full.)

In POWERplay, multi-agent MDPs are called **joint MDPs** because they assume a **joint state space** and a **joint action space** between the two agents.

A **state** in a joint MDP describes the joint positions of Agent H and Agent A. For example, `'(1, 3)_H^(0, 2)_A'`, at the bottom left of the MDP graph above, means that Agent H is at position `(1, 3)` on the grid, and Agent A is at position `(0, 2)` on the grid.

An **action** in a joint MDP describes a pair of actions Agent H and Agent A can jointly take from a given state. For example, from the `'(1, 3)_H^(0, 2)_A'` state at the bottom left of the MDP graph above, one of the actions is `'left_H^down_A'`. This means that Agent H takes the `'left'` action from this state, while Agent A simultaneously takes the `'down'` action.

As you might imagine, multi-agent MDPs can quickly get large and complicated. Because the joint state space of a multi-agent MDP is the outer product of its single-agent state space, it grows quadratically with the number of single-agent states. So a gridworld with 30 cells has 30 x 30 = 900 multi-agent states. (And the same reasoning holds for the joint action space.)

POWERplay can scale to support about 1600 joint states on a modern laptop (e.g., M1 MacBook Pro), which enables experiments on multi-agent gridworlds with about 40 cells. This is big enough to let you investigate a number of non-trivial multi-agent behaviors and interaction types.

🔵 Here are the input arguments to `base.construct_multiagent_gridworld_mdp()` and what they mean:

(Listed as `name [type] (default): description`.)

- `num_rows [int, required]`: The maximum number of rows in your gridworld.

- `num_cols [int, required]`: The maximum number of columns in your gridworld.

  Typical value: `4`

- `squares_to_delete [list] ([])`: A list of 2-tuples, where each 2-tuple is a pair of coordinates (in **string** format) that represent the edges of a square you want to delete from your gridworld. For example, if you want to delete the square with the top-left corner at (0, 0) and the bottom-right corner at (2, 2), then you would use `squares_to_delete=[['(0, 0)', '(2, 2)']]`. This format allows us to quickly construct gridworlds with interesting structures.

  Typical value: `[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']]`

🟢 Here is the output to `base.construct_multiagent_gridworld_mdp()`:

- `mdp_graph [networkx.DiGraph]`: An MDP graph representing the multi-agent gridworld you created. Each square has a self-loop, and connections to the squares above, beneath, to the left, and to the right of it (if they exist).

  The states of a multi-agent gridworld MDP are strings that indicate where Agent H and Agent A are located in the gridworld. For example, the state `'(0, 0)_H^(0, 1)_A'` means that Agent H is located at position `(0, 0)` (top left), and Agent A is located at position `(0, 1)` (top, one cell from the left).

  The actions of a multi-agent gridworld MDP are strings that indicate the allowed actions that Agent H and Agent A can jointly take from each state. For example, `'left_H^stay_A'` means the action where Agent H moves left and Agent A stays where it is. If one or both of the agents are located at an edge or corner of the gridworld, some of these joint actions will be forbidden.

  You can save your `mdp_graph` with `base.save_mdp_graph()`, view it as a gridworld with `base.view_gridworld()`, and view its full MDP with `base.plot_mdp_or_policy()`.

#### Edit a gridworld MDP

MDPs can get complicated, and multi-agent MDPs can get complicated especially quickly. It's possible to edit your MDPs manually using functions in the `mdp` and `multi` modules, but this can often be tedious and error-prone unless you write your own wrapper functions to accelerate the process.

POWERplay comes with a text-based MDP editor that lets you edit single-agent and multi-agent MDPs you've created. Its capabilities are limited, but it makes it slightly easier to get started if you want to define complicated dynamics for your agents.

🟣 To access the editor, first define an MDP you want to edit, then call `base.edit_gridworld_mdp()`. For example, for a single-agent MDP:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> edited_gridworld = base.edit_gridworld_mdp(gridworld)
```

Follow the instructions as you're prompted in the command line. For each state, the prompt will tell you which actions are currently allowed from that state (e.g., `'down'`, `'stay'`). It will ask you to input new allowed actions from that state (so if, e.g., you input `'down'`, only the `'down'` action will be allowed and the `'stay'` action will be forbidden from that state). If you skip one of these prompts, the set of actions is unchanged.

If you input actions, you'll be asked to input the states that those actions lead to. Again, if you skip one of these prompts, the set of actions is unchanged.

Here's an example of using the MDP editor to delete the `'stay'` action from the `'(0, 2)'` state of the 3x4 gridworld maze MDP:

```
>>> base.plot_mdp_or_policy(edited_gridworld)
```

![3x4 gridworld maze MDP example with 'stay' action deleted from (0, 2) state](img/3x4-gridworld-maze-02-stay-deleted-example-mdp.png)

Notice that only the `'down'` action remains from that state, so the agent is forced to move down whenever it lands on the `'(0, 2)'` state.

The MDP editor also works on multi-agent gridworlds. For example:

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> edited_multiagent_gridworld = base.edit_gridworld_mdp(multiagent_gridworld)
```

With a multi-agent gridworld, you have the option of deleting overlapping states in the MDP. This is equivalent to forbidding your agents from occupying the same cell on the gridworld. It's a kind of [physical interaction](https://www.alignmentforum.org/posts/nisaAr7wMDiMLc2so/instrumental-convergence-scale-and-physical-interactions#3_2_Physically_interacting_agents) you can impose on your agents.

The rest of the interface works similarly in the multi-agent case as it does in the single-agent case. The only difference is that the state and action spaces are now **joint** state and action spaces. So instead of states like `'(0, 2)'`, you'll see states like `'(0, 2)_H^(1, 2)_A'` since both agents' positions need to be defined by the state. (And similarly for the actions.)

Here's an example of using the MDP editor to delete all the overlapping states from the 3x4 gridworld maze MDP:

```
>>> base.plot_mdp_or_policy(edited_multiagent_gridworld)
```

![3x4 multi-agent gridworld maze example with overlapping states deleted](img/3x4-multiagent-gridworld-maze-overlapping-states-deleted-example-mdp.png)

You'll notice that there are no states like `'(0, 2)_H^(0, 2)_A'` in this MDP; i.e., Agent H and Agent A have been forbidden from occupying the same positions.

🔵 Here are the input arguments to `base.edit_gridworld_mdp()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.DiGraph, required]`: An MDP graph. This can be a single-agent MDP graph (if it's the output of `base.construct_single_agent_gridworld_mdp()`) or a multi-agent MDP graph (if it's the output of `base.construct_multiagent_gridworld_mdp()`).

  Typical value: `base.construct_single_agent_gridworld_mdp(3, 4)`

🟢 Here is the output to `base.edit_gridworld_mdp()`:

- `output_mdp_graph [networkx.DiGraph]`: An edited MDP graph. This can be a single-agent or multi-agent MDP graph, depending on which kind of graph you used as input.

  You can save your `output_mdp_graph` with `base.save_mdp_graph()`, view it as a gridworld with `base.view_gridworld()`, and view its full MDP with `base.plot_mdp_or_policy()`.

#### Construct a policy graph

If you're running a multi-agent experiment in POWERplay, you'll need to define a policy graph for **Agent A**. This will either be a policy graph that Agent A always follows, or a ["seed policy"](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#A_1_Initial_optimal_policies_of_Agent_H) that Agent A is initialized at before learning different policies.

We encode policy graphs in the same format as MDP graphs. Importantly, **a policy graph is always tied to a _specific_ MDP graph**. This is because a policy defines all the actions an agent can take from every state in an MDP graph. So a policy always has to exactly reflect the names of the states and actions, and the topology, of the MDP graph it's associated with.

🟣 For this reason, you always create a policy _from_ an MDP graph, using `base.construct_policy_from_mdp()`. For example:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> random_policy = base.construct_policy_from_mdp(gridworld)
>>> base.plot_mdp_or_policy(random_policy)
```

![3x4 gridworld maze policy example](img/3x4-gridworld-maze-policy-example.png)

The function `base.construct_policy_from_mdp()` will always generate a uniform random policy, meaning that at each state in its MDP, the policy has an equal probability of taking each legal action.

The above example creates a policy on a single-agent MDP, which is something you'll rarely need to do. More often, you'll be creating a policy for Agent A from a joint multi-agent MDP. For example:

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> random_policy_agent_A = base.construct_policy_from_mdp(multiagent_gridworld, acting_agent_is_H=False)
>>> base.plot_mdp_or_policy(random_policy_agent_A)
```

![3x4 multi-agent gridworld maze policy example](img/3x4-multiagent-gridworld-maze-policy-example.png)

Notice in the above that while the policy has a **joint state space** (i.e., its states are of the form `'(2, 0)_H^(1, 2)_A'`), its **action space** consists of only the actions Agent A can take. This is because we set `acting_agent_is_H=False`, which makes Agent A (not Agent H) the agent that acts out the policy.

🔵 Here are the input arguments to `base.construct_policy_from_mdp()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.DiGraph, required]`: The MDP graph you want to construct the policy on. Can by either a single-agent or joint multi-agent MDP graph.

  Typical value: `base.construct_multiagent_gridworld_mdp(3, 4)`

- `acting_agent_is_H [bool] (False)`: A Boolean flag that indicates which agent you want to define the policy for. If `mdp_graph` is a single-agent MDP, this flag has no effect. If `mdp_graph` is a joint multi-agent MDP, setting this to `True` will output a policy for Agent H. Keeping this `False` will output a policy for Agent A, which is the more common case.

  Typical value: `False`

🟢 Here is the output to `base.construct_policy_from_mdp()`:

- `policy_graph [networkx.DiGraph]`: A policy graph that's compatible with the MDP `mdp_graph`. This will always be a uniform random policy, meaning that an agent that follows the policy will always select its action randomly at every state, with equal probability over all the actions that are allowed at that state.

  If `mdp_graph` is a single-agent MDP, `policy_graph` will represent the policy of Agent H (the only agent). If `mdp_graph` is a joint multi-agent MDP, `policy_graph` will represent the policy of Agent H if `acting_agent_is_H=True`, and the policy of Agent A if `acting_agent_is_H=False`.

  You can save your `policy_graph` with `base.save_policy_graph()`, and view it with `base.plot_mdp_or_policy()`.

#### Edit a gridworld policy graph

🟣 Just like for MDPs, POWERplay includes a text based interface you can use to edit gridworld policy graphs. To use it, first create a policy, and then call `base.edit_gridworld_policy()`:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> random_policy = base.construct_policy_from_mdp(gridworld)
>>> edited_policy = base.edit_gridworld_policy(random_policy)
```

Follow the instructions as you're prompted in the command line. For each state, the prompt will ask you which action you want your policy to take at that state. You have to pick an action that's currently allowed at that state (e.g., if the state is in the top-left of the grid, you won't be allowed to pick `'up'`). If you skip the prompt for one of the states, the set of action probabilities for that state is unchanged.

Here's an example of using the policy editor to create a policy that always stays where it is on the 3x4 gridworld maze MDP:

```
>>> base.plot_mdp_or_policy(edited_policy)
```

![3x4 gridworld maze stay policy example](img/3x4-gridworld-maze-policy-stay-example.png)

In the above, the agent chooses the `'stay'` action with probability 1 at every state.

You can also use `base.edit_gridworld_policy()` to edit policies defined on joint multi-agent MDPs:

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> random_policy_agent_A = base.construct_policy_from_mdp(multiagent_gridworld, acting_agent_is_H=False)
>>> edited_policy_agent_A = base.edit_gridworld_policy(random_policy_agent_A)
```

Here's an example of using the policy editor to create an Agent A policy that chooses the `'stay'` action at state `'(0, 2)_H^(0, 2)_A'`, but is otherwise random:

```
>>> base.plot_mdp_or_policy(edited_policy_agent_A)
```

![3x4 multi-agent gridworld maze policy that stays at state '(0, 2)_H^(0, 2)_A' and otherwise moves randomly](img/3x4-gridworld-maze-policy-02-02-stay-example.png)

🔵 Here are the input arguments to `base.edit_gridworld_policy()` and what they mean:

(Listed as `name [type] (default): description`.)

- `policy_graph [networkx.DiGraph, required]`: A policy graph. This can represent a policy defined on either a single-agent or a joint multi-agent MDP.

  Typical value: `base.construct_policy_from_mdp(base.construct_single_agent_gridworld_mdp(3, 4))`

🟢 Here is the output to `base.edit_gridworld_policy()`:

- `output_policy_graph [networkx.DiGraph]`: An edited policy graph. This can represent a policy defined on either a single-agent or a joint multi-agent MDP, depending on which kind you used as input.

  You can save your `output_policy_graph` with `base.save_policy_graph()`, and view it with `base.plot_mdp_or_policy()`.

#### Convert a single-agent policy to a multi-agent policy

One big challenge with manually editing multi-agent policies is that it can take a very long time, because multi-agent MDPs have a lot of joint states to define actions on. A good workflow to speed this up is to:

1. Define a single-agent policy (with few states)
2. Edit that single-agent policy manually
3. **Convert** your edited single-agent policy to a multi-agent policy
4. Make final edits to the multi-agent policy manually

🟣 You can do step 3 using `base.single_agent_to_multiagent_policy()`. The whole workflow might look like this:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> random_policy = base.construct_policy_from_mdp(gridworld)
>>> edited_policy = base.edit_gridworld_policy(random_policy)

[... edits ...]

>>> multiagent_policy = base.single_agent_to_multiagent_policy(edited_policy, acting_agent_is_H=False)
>>> edited_multiagent_policy = base.edit_gridworld_policy(multiagent_policy)
```

If you don't need to construct a policy that has actions conditional on the other agent's state, you can skip step 4 of this workflow. But in either case, you'll need to manually define a multi-agent MDP that will be **paired** with your edited multi-agent policy:

```
multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
```

Here's an example of using this workflow to define a multi-agent policy that chooses the `'stay'` action everywhere, except at state `'(0, 2)_H^(0, 2)_A'`, where it chooses the `'down'` action:

```
>>> base.plot_mdp_or_policy(edited_multiagent_policy)
```

![3x4 multi-agent gridworld maze policy that moves down at state '(0, 2)_H^(0, 2)_A' and otherwise stays where it is](img/3x4-multiagent-gridworld-maze-policy-02-02-down-else-stay-example.png)

🔵 Here are the input arguments to `base.single_agent_to_multiagent_policy()` and what they mean:

(Listed as `name [type] (default): description`.)

- `single_agent_policy_graph [networkx.DiGraph, required]`: A policy graph defined on a single-agent MDP.

  Typical value: `base.construct_policy_from_mdp(base.construct_single_agent_gridworld_mdp(3, 4))`

- `acting_agent_is_H [bool] (False)`: A Boolean flag that indicates which agent you want to define the output multi-agent policy for. Setting this to `True` will output a policy for Agent H. Keeping this `False` will output a policy for Agent A, which is the more common case.

🟢 Here is the output to `base.single_agent_to_multiagent_policy()`:

- `multiagent_policy_graph [networkx.DiGraph]`: A multi-agent policy graph based on the input `single_agent_policy_graph`. If `acting_agent_is_H=False`, this will be a policy graph for Agent A that takes the actions consistent with `single_agent_policy_graph`. For example, if `single_agent_policy_graph` takes the action `'stay'` from state `'(0, 0)'`, then `multiagent_policy_graph` will take the action `'stay'` from states `'(0, 0)_H^(0, 0)_A'`, `'(0, 1)_H^(0, 0)_A'`, and any other state in which Agent A is located at `(0, 0)`. (And similarly for `acting_agent_is_H=True` in the Agent H case.)

  You can save your `multiagent_policy_graph` with `base.save_policy_graph()`, and view it with `base.plot_mdp_or_policy()`.

#### View a gridworld

 🟣 You can view a gridworld MDP by using the `base.view_gridworld()` function. Here's what a tiny single-agent gridworld maze looks like, for example:

 ```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> base.view_gridworld(gridworld)
```

![3x4 gridworld maze example](img/3x4-gridworld-maze-example.png)

For a single-agent gridworld, `base.view_gridworld()` shows a single plot with the coordinates of each gridworld cell on the x- and y-axes of the plot. The caption of this plot, and the numbers inside the cells, are meaningless. The `base.view_gridworld()` function is best used for quick visualizations and troubleshooting.

You can use `base.view_gridworld()` to view multi-agent gridworlds as well. Because multi-agent gridworlds are defined on a **joint state space** (i.e., over the positions of both Agent H and Agent A), the visualization is more complicated. Here's an example of the same gridworld as above, but in multi-agent format:

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> base.view_gridworld(multiagent_gridworld)
```

![3x4 multi-agent gridworld maze example](img/3x4-multiagent-gridworld-maze-example.png)

This time, `base.view_gridworld()` produces many plots, one for each position of Agent A. Within each plot, each cell corresponds to a position of Agent H, and the red open square shows the position of Agent A within that block.

🔵 Here are the input arguments to `base.view_gridworld()` and what they mean:

(Listed as `name [type] (default): description`.)

- `gridworld_mdp_graph [networkx.DiGraph, required]`: A gridworld MDP graph. Can be either a single-agent MDP or a joint multi-agent MDP, but it has to be formatted as a gridworld.

  Typical value: `base.construct_multiagent_gridworld_mdp(3, 4)`

#### Plot an MDP or policy

🟣 You can plot any kind of MDP or policy — single-agent or multi-agent — using `base.plot_mdp_or_policy()`. For example, here's how to plot a single-agent random policy on a tiny maze gridworld:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> random_policy = base.construct_policy_from_mdp(gridworld)
>>> base.plot_mdp_or_policy(random_policy)
```

![3x4 gridworld maze policy example](img/3x4-gridworld-maze-policy-example.png)

And here's how to plot the multi-agent MDP that corresponds to the same gridworld:

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> base.plot_mdp_or_policy(multiagent_gridworld)
```

![3x4 multi-agent gridworld maze MDP example](img/3x4-multiagent-gridworld-maze-example-mdp.png)

The `base.plot_mdp_or_policy()` function plots 4 "state trees" per row by default, as you can see from the two graphs above. Sometimes joint multi-agent MDPs have so many states that it's impractical to plot all of them in a single figure. POWERplay can chunk these big plots over multiple different figures if that happens.

🔵 Here are the input arguments to `base.view_gridworld()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_or_policy_graph [networkx.DiGraph, required]`: An MDP or policy graph. Can be either a single-agent or multi-agent in either case.

  Typical value: `base.construct_multiagent_gridworld_mdp(3, 4)`

- `subgraphs_per_row [int] (4)`: The number of state subgraphs to plot per row of the final figure.

  Typical value: `4`

- `subgraphs_per_figure [int] (128)`: The number of state subgraphs to plot per figure. If the graph you're plotting has more states than this, the state subgraphs will get chunked over multiple figures. For example, if you're plotting an MDP with 300 states and `subgraphs_per_figure=128`, POWERplay will plot your MDP as two figures with 128 subgraphs, and one figure with (300 - 2 x 128) = 44 subgraphs.

#### Save an MDP graph

🟣 To save an MDP graph you've created or edited, use `base.save_mdp_graph()`. For example:

```
>>> gridworld = base.construct_single_agent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> base.save_mdp_graph(gridworld, '3x4_gridworld_maze')
```

This will save your MDP in the `mdps/` folder of the POWERplay directory. If you're saving a multi-agent MDP graph, **best practice** is to prefix `'joint_'` to the filename, e.g.,

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> base.save_mdp_graph(multiagent_gridworld, 'joint_3x4_gridworld_maze')
```

🔵 Here are the input arguments to `base.save_mdp_graph()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_graph [networkx.DiGraph, required]`: The MDP graph you want to save.

  Typical value: `base.construct_multiagent_gridworld_mdp(3, 4)`

- `mdp_filename [str, required]`: The filename under which to save the MDP graph in the `mdps/` folder. If you're saving a multi-agent MDP graph, prefix the filename with `'joint_'`.

  Typical value: `'joint_3x4_gridworld_maze'`

#### Load an MDP graph

🟣 To load an MDP graph into memory, use `base.load_mdp_graph()`. For example:

```
>>> multiagent_gridworld = base.load_mdp_graph('joint_3x3_gridworld')
```

You can then edit the MDP graph just like any other.

🔵 Here are the input arguments to `base.load_mdp_graph()` and what they mean:

(Listed as `name [type] (default): description`.)

- `mdp_filename [str, required]`: The filename of the MDP graph you want to load, not including the extension. You'll find existing MDP graphs in the `mdps/` folder.

  Typical value: `'joint_3x3_gridworld'`

#### Save a policy graph

🟣 To save a policy graph you've created or edited, use `base.save_policy_graph()`. For example:

```
>>> multiagent_gridworld = base.construct_multiagent_gridworld_mdp(3, 4, squares_to_delete=[['(0, 0)', '(1, 1)'], ['(0, 3)', '(0, 3)'], ['(2, 3)', '(2, 3)']])
>>> random_policy_agent_A = base.construct_policy_from_mdp(gridworld, acting_agent_is_H=False)
>>> base.save_policy_graph(random_policy_agent_A, 'joint_3x4_gridworld_maze_agent_A_uniform_random')
```

This will save your policy in the `policies/` folder of the POWERplay directory. When saving a policy, **best practice** is to prefix the policy filename with the _entire filename_ of the MDP it's associated with, then the agent the policy is for, and finally a description of what the policy does. In the example above, `'joint_3x4_gridworld_maze'` is the filename of the MDP the policy is defined on; `'agent_A'` is the agent the policy is for, and `'uniform_random'` says that the policy is a uniform random policy.

🔵 Here are the input arguments to `base.save_policy_graph()` and what they mean:

(Listed as `name [type] (default): description`.)

- `policy_graph [networkx.DiGraph, required]`: The policy graph you want to save.

  Typical value: `base.construct_multiagent_gridworld_mdp(3, 4)`

- `policy_filename [str, required]`: The filename under which to save the policy graph in the `policies/` folder. Always prefix the policy filename with the _entire filename_ of the MDP it's associated with, then the agent the policy is for (H or A), and finally a description of what the policy does.

  Typical value: `'joint_3x4_gridworld_maze_agent_A_uniform_random'`

#### Load a policy graph

🟣 To load a policy graph into memory, use `base.load_policy_graph()`. For example:

```
>>> random_policy = base.load_policy_graph('joint_3x3_gridworld_agent_A_uniform_random')
```

You can then edit the policy graph just like any other.

🔵 Here are the input arguments to `base.load_policy_graph()` and what they mean:

(Listed as `name [type] (default): description`.)

- `policy_filename [str, required]`: The filename of the policy graph you want to load, not including the extension. You'll find existing MDP graphs in the `policies/` folder.

  Typical value: `'joint_3x3_gridworld_agent_A_uniform_random'`

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

### Creating a gridworld MDP graph

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
