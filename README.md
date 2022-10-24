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

    ![Example of wandb sweep data](img/wandb-sweep-data-example.png)

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
>>> base.launch_experiment('my_first_sweep.yaml', plot_as_gridworld=True)
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

This function executes the sweep defined in the config file `configs/test/test_vanilla.yaml`. The outputs of the sweep, incluing figures, get saved locally under `expts/{SWEEP_ID}-test_vanilla`. (And get separately uploaded to wandb.)

#### Gridworld test

🟣 If you want to test POWERplay's ability to plot POWERs on gridworlds, use `base.test_gridworld()`. Instead of displaying POWERs in a bar plot, this function displays them as a heat map on a gridworld.

For example:

```
>>> base.test_gridworld()
```

This function executes the sweep defined in the config file `configs/test/test_gridworld.yaml`. The agent's environment is a simple 3x3 gridworld.

The outputs of the sweep get uploaded to wandb and separately get saved locally under `expts/{SWEEP_ID}-test_gridworld`. Files with the prefix `POWER_means-agent_H` are figures that display POWERs as gridworld heatmaps.

#### Stochastic test

🟣 If you want to test POWERplay's ability to compute POWERs on _stochastic_ MDPs, use `base.test_stochastic()`.

For example:

```
>>> base.test_stochastic()
```

This function executes the sweep defined in the config file `configs/test/test_stochastic.yaml`. The agent's environment is a 4-state MDP with stochastic transitions.

The outputs of the sweep get uploaded to wandb and separately get saved locally under `expts/{SWEEP_ID}-test_stochastic`.

#### Fixed-policy multi-agent test

🟣 If you want to test POWERplay's ability to compute POWERs on multi-agent MDPs, use `base.test_multiagent()`. Note that this computes Agent H POWERs assuming a _fixed policy_ for Agent A. So this test isn't "truly" multi-agent, because we can just _simulate_ Agent A's presence by incorporating its dynamics into Agent H's perceived transition function.

For example:

```
>>> base.test_multiagent()
```

This function executes two sweeps. The first, defined in the config file `configs/test/test_multiagent_simulated.yaml`, _simulates_ the presence of Agent A, but does not execute any multi-agent code. The second sweep, defined in the config file `configs/test/test_multiagent_actual.yaml`, runs Agent A using a defined fixed policy and multi-agent code. If POWERplay is functioning correctly, the outputs of these two sweeps should be exactly identical.

The outputs of these sweep get uploaded to wandb and separately get saved locally under `expts/{SWEEP_ID}-test_multiagent_simulated` and `expts/{SWEEP_ID}-test_multiagent_actual`.

#### Reward-correlation multi-agent test

🟣 To test POWERplay's ability to compute POWERs on multi-agent MDPs [where each agent may have a different reward function](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#3_1_Multi_agent_reward_function_distributions), use `base.test_reward_correlation()`. This test function uses a [novel definition of POWER](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2__Multi_agent_POWER__human_AI_scenario) (i.e., [instrumental value](#what-can-i-do-with-it)) that applies to certain important kinds of multi-agent systems.

For example:

```
>>> base.test_reward_correlation()
```

This function executes the sweep defined in the config file `configs/test/test_reward_correlation.yaml`. The agents' environment is a 3x3 gridworld with joint states over the positions of the two agents — so 81 states in total, since either agent can occupy any of 9 positions on the 3x3 grid.

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

🟢 Here is the output to `base.load_mdp_graph()`:

- `mdp_graph [networkx.DiGraph]`: The MDP graph you've loaded. Can be single-agent or multi-agent.

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

🟢 Here is the output to `base.load_policy_graph()`:

- `policy_graph [networkx.DiGraph]`: The policy graph you've loaded. Can be single-agent or multi-agent.

### Launching an experiment

🟣 To launch an experiment, use the `base.launch_experiment()` function. This function takes a config filename as its only required argument. The config file is a YAML file that contains all the parameters for the experiment. For example:

```
>>> base.launch_experiment('my_first_sweep.yaml', plot_as_gridworld=True, plot_distributions=True, plot_correlations=True)
```

Config files should be located in the `configs/` directory. There are also some example config files in `configs/test/` and `configs/replication/`.

The config file is the canonical description of your whole experiment. It usually defines a **sweep** over parameters that is made up of individual **runs**. During an experiment, the YAML files that correspond to each run of your sweep are saved in the `expts/` and `wandb/` directories. This ensures that even if you edit the YAML file in `configs/`, you'll always know what parameters you used for every run. This ensures your experiments will always be reproducible.

🔵 Here are the input arguments to `launch_experiment()` and what they mean (the YAML API is described afterwards):

(Listed as `name [type] (default): description`.)

- `config_filename [str, required]`: The name of the YAML file that contains the configuration for your experiment sweep, including the `'.yaml'` extension. This file should be located in the `configs/` directory.

  Typical value: `'my_first_sweep.yaml'`

- `wandb_entity [str] (data.get_settings_value(data.WANDB_ENTITY_PATH, settings_filename=data.SETTINGS_FILENAME))`: The wandb entity that is running the sweep. This is usually your wandb username, but it can sometimes be a team if you're collaborating with someone else. This is used to properly save your sweep in wandb.

  Typical value: `'bob-bobson'`

- `wandb_project [str] ('uncategorized')`: The  project that's running your experiment sweep. This is used to properly save your sweep in wandb. If a project of that name is _not_ owned by the same `wandb_entity` that is running the sweep, a new project will be created.

  Typical value: `'uncategorized'`

- `number_of_workers [int | None] (None)`: The number of CPU workers to run your experiment on. Since the POWER calculation is embarrassingly parallel, POWERplay is designed to scale it gracefully across processes using [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html). POWERplay isn't configured to run on the cloud, but it should at least be able to make full use of all your local CPU resources.

  If `number_of_workers=None`, POWERplay automatically sets it to the maximum number of available CPU workers on your machine using `multiprocessing.cpu_count()`. This is fine in the vast majority of cases.

  Typical value: `10`

- `plot_as_gridworld [bool] (False)`: POWERplay automatically generates figures of your results during a sweep, and saves them to the `expts/` folder and to wandb. If the MDP in your experiment is a gridworld, setting `plot_as_gridworld=True` will format these output plots in a way that's especially convenient to visualize. If you're working with gridworlds, you should always manually set this to `True`.

  If the MDP in your experiment _isn't_ a gridworld, setting this to `plot_as_gridworld=True` will crash the run. This is why its default value is `False`.

  Typical value: `True`

- `plot_distributions [bool] (False)`: POWERplay calculates POWERs by sampling a large number of reward functions (typically 10,000 or more) from a reward distribution. It then computes the [average optimal value](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems#2_1_Definition) at each state, for each of those sampled reward functions, and then averages them together to get its final estimate of the POWER value. Each of the average optimal values POWERplay calculates is called a **"POWER sample"**. So the POWER value is the average of all the POWER samples.

  If you set `plot_distributions=True`, POWERplay will automatically generate histograms of the reward values and POWER samples at every state in your MDP. This is extremely useful for troubleshooting, debugging, or understanding complicated or unexpected behaviors. But because this option generates one histogram for every state of your MDP, it can quickly get unwieldy for a non-trivial multi-agent MDP with hundreds or thousands of states.

  Generally best practice is to leave `plot_distributions=False` if you're working with a multi-agent system. You can set `plot_distributions=True` at small scale for testing and debugging, or if there's a simple system that you're trying to understand very deeply.

  Typical value: `False`

- `plot_correlations [bool] (False)`: If you set `plot_correlations=True`, POWERplay will automatically generate 2D histograms showing how the POWER samples at each state of your MDP correlates with the POWER samples at every other state of your MDP. This is even more informative than setting `plot_distributions=True`, because it can reveal deep relationships between the instrumental values of different states.

  Unfortunately, plotting correlations is even more intensive than plotting distributions, because it generates a histogram for every possible _pair_ of states in your MDP. So if your MDP has `N` states, setting `plot_distributions=True` will generate `N**2` histograms. This is hugely impractical for any MDP with more than a couple dozen states, which in practice rules out multi-agent systems completely. Still, this option is occasionally worth using when troubleshooting small single-agent systems.

  Typical value: `False`

- `diagnostic_mode [bool] (False)`: Only used for experiments of type `'multiagent_with_reward'` (see below), that use [joint reward function distributions](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#3_1_Multi_agent_reward_function_distributions). If `True`, this will plot the POWERs of the [Agent A seed policy](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#A_1_Initial_optimal_policies_of_Agent_H), _and_ plot the POWERs of Agent H against the Agent A seed policy. The latter, in particular, is often interesting to visualize because it shows [how well Agent H does against "nature"](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2__Multi_agent_POWER__human_AI_scenario) in the absence of optimization from Agent A. This makes it possible to investigate questions like, "at what point should the human agent _prefer_ an AI-dominated world, to a world without a powerful AI?"

  We generally recommend leaving this set to `True` for `'multiagent_with_reward'` experiments. For other experiment types, this setting has no effect.

  Typical value: `True`

#### Experiment config files

You can find examples of experiment configuration files in the `configs/test/` and `configs/replication/` folders. There are three types of experiment that POWERplay recognizes:

1. `'single_agent'`: A single-agent MDP from the perspective of Agent H. POWER calculations are based on the [single-agent definition](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems#2_1_Definition). Example config file: `configs/test/test_stochastic.yaml`.

2. `'multiagent_fixed_policy'`: A multi-agent MDP from the perspective of Agent H, where Agent A runs a fixed policy. This is conceptually identical to the `'single_agent'` type, since from Agent H's perspective, Agent A can be thought of as part of a fixed MDP. In fact, POWER calculations are based on the [single-agent definition](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems#2_1_Definition) here as well. But explicitly defining a fixed Agent A policy can make this situation easier to reason about. Example config file: `configs/test/test_multiagent_actual.yaml`.

3. `'multiagent_with_reward'`: A multi-agent MDP in which Agent H learns optimal policies against a fixed environment, and Agent A then learns optimal policies against fixed Agent H policies. POWER calculations are based on the [multi-agent definitions described here](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2__Multi_agent_POWER__human_AI_scenario). Example config file: `configs/test/test_reward_correlation.yaml`.

POWERplay auto-detects the experiment type based on which `parameters` are defined in the config file.

Here are the entries you need in your experiment config file:

- `name [str]`: The name of your experiment, typically separated by underscores. This should be the same as the name of the YAML config file.

  Typical value: `'test_vanilla'`

- `description [str]`: A description of your experiment.

  Typical value: `'Sweeping over discount rates on the MDP from Figure 1 in Optimal Policies Tend to Seek Power, https://arxiv.org/abs/1912.01683'`

- `program`: The Python script that runs your experiment. This should always be `'sweep.py'`.

  Typical value: `'sweep.py'`

- `parameters`: A series of entries and values that define the sweep. Includes fixed values and values that vary over the sweep. Below are the parameters and their syntax **for fixed values** (i.e., values that stay constant over a sweep):

  - `discount_rate`: The discount rate for Agent H. If the experiment type is `'single_agent'`, this is the discount rate for the single agent. Should be a number between 0 and 1.

    Typical value: `0.6`
  
  - `reward_distribution`: A distribution of reward functions for Agent H over which POWERplay will calculate POWER. POWERplay only supports reward functions distributions with rewards that are [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) over MDP states. In other words, for each reward function sample, the reward value at one state is independent of the reward value at all the other states of the MDP.
  
      This simplification makes reward function distributions in POWERplay a bit easier to define.
      
      - We first define a `default_dist`, which is a reward distribution that applies over all the states of the MDP. So for example, if we defined the `default_dist` as a Uniform(0, 1) distribution, then POWERplay would sample a reward function by sampling reward values from Uniform(0, 1), independently at every state. The `default_dist` entry for a Uniform(0, 1) distribution looks like this:

        ```
        default_dist:
          dist_name:
            "uniform"
          params:
            [0, 1]
        ```

        As you can see, to define a reward distribution on a state, we need two things. First is the `dist_name`, which in this case is `"uniform"` for the uniform distribution. You can find the list of supported `dist_name`s in the module `src/lib/utils/dist.py` under the `DISTRIBUTION_DICT` variable. These are typically `torch.distribution` objects, but POWERplay also supports manually defining arbitrary reward distributions.

        The second thing you need to define a distribution is the distribution's `params`. These are the parameters of the distribution, and they're different for each distribution. In the `"uniform"` example above, we're using the `torch.distributions.Uniform` distribution, so the two parameters define the [support](https://en.wikipedia.org/wiki/Support_(mathematics)) of the uniform distribution. In other words, `params: [0, 1]` in this case means a uniform distribution from 0 to 1. But if we had `dist_name: "beta"`, then `params` would refer to the [alpha and beta parameters of the beta distribution](https://en.wikipedia.org/wiki/Beta_distribution).
      
      - The second thing we define is a list of `state_dists`, which is alloweds to be empty. The entry `state_dists` is a _list_ of distributions that apply to specific states of the MDP, and that override the `default_dist` on the states they're associated with. If we want to have a "special" state whose rewards are, say, always fixed at zero over the reward function distribution, we can get that using `state_dists`. In POWERplay, every state in an MDP is labeled with a unique string, and `state_dists` lets you assign reward distributions to individual states using their string labels. Here's an example of a `state_dists` entry that defines special distributions for two MDP states labeled `"∅"` and `"ℓ_◁"`:

        ```
        state_dists:
        "∅":
          dist_name:
            "uniform"
          params:
            [-1, 1]
        "ℓ_◁":
          dist_name:
            "beta"
          params:
            [1, 2]
        ```
    
      - Finally, we define an `allow_all_equal_rewards` Boolean parameter. If set to `False`, POWERplay will re-sample a reward function if it encounters a reward function whose rewards are exactly equal everywhere. This sometimes happens when using discrete reward function distributions (e.g., `dist_name: "bernoulli"`, on which a state's reward is always exactly 0 or 1). If a sampled reward function is 0 everywhere, for example, then the agent's optimal policy will be arbitrary and this can affect the results of a POWER calculation.

        On continuous reward function distributions, this almost never happens, so you can set `allow_all_equal_rewards: True`. This is the typical case.
    
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
          "beta"
        params:
          [1, 2]
    allow_all_equal_rewards:
      True
    ```
  
  - `num_reward_samples`: The number of samples to draw from the `reward_distribution` when calculating POWERs. POWERplay parallelizes its calculation across `num_reward_samples`, so if you have lots of CPUs available you can run more reward samples.

    Setting this to a high value (e.g., `100000`) makes POWER calculations more accurate at the expense of taking longer to do. For big multi-agent systems, anything between `10000` and `20000` is pretty typical.

    **NOTE:** `num_reward_samples` should be divisible by `number_of_workers` (the input to `base.launch_experiment()`) for perfectly reproducible results. Otherwise, POWERplay will truncate your reward samples until they can be divided evenly between all your workers. So if you have 3 workers and want to run 100 reward samples, POWERplay will only actually run 99 reward samples (giving each worker 33 samples to run).

    Typical value: `10000`
  
  - `convergence_threshold`: The convergence threshold for the value iteration algorithm. You generally never need to change this from the default value.

    Typical value: `0.0001`
  
  - `random_seed`: The random seed to use for the experiment. If `null`, then the random seed is not fixed. We recommend setting this to a number, to make it easier to reproduce your experiment later if you need to.

    Typical value: `0`

  The `parameters` above need to be defined for all experiment types. But some `parameters` are specific to particular experiment types. POWERplay auto-detects the experiment type based on which `parameters` are defined in the experiment config file, and throws an error if the set of `parameters` you've defined is insconsistent with any known experiment type.

  Below are the `parameters` you need to define for the `'single_agent'` experiment type. You can find an example config file for the `'single_agent'` experiment type under `configs/test/test_stochastic.yaml`.

    - `mdp_graph`: The name of the file in the `mdps/` folder that corresponds to the MDP graph you want to use. This name should not include the `.gv` extension.
    
      This should be a **single-agent** MDP graph, not a joint multi-agent graph.

      Typical value: `"3x3_gridworld"`
  
  Here are the `parameters` you need to define for the `'multiagent_fixed_policy'` experiment type. You can find an example config file for the `'multiagent_fixed_policy'` experiment type under `configs/test/test_multiagent_actual.yaml`.

    - `joint_mdp_graph`: The name of the file in the `mdps/` folder that corresponds to the joint MDP graph you want to use for your two agents. This name should not include the `.gv` extension.
    
      This should be a **joint multi-agent** MDP graph, not a single-agent graph. Best practice is to add the prefix `'joint_'` to all joint MDP graph filenames when [saving them](#save-an-mdp-graph), so it should be easy to tell which type of graph you're working with.

      Typical value: `"joint_3x3_gridworld"`

    - `policy_graph_agent_A`: The name of the file in the `policies/` folder that corresponds to the policy graph you want to assign to Agent A. This name should not include the `.gv` extension. POWERplay will calculate the POWERs of Agent H assuming Agent A always follows this fixed policy.

      This policy graph should be **compatible** with `joint_mdp_graph` above, in the sense that its sets of states and actions should be the same as those of `joint_mdp_graph`. Best practice is to prefix a policy filename with the _entire filename_ of the MDP it's associated with when [saving a policy](#save-a-policy-graph). So it should be easy to tell from the filenames whether `policy_graph_agent_A` is compatible with `joint_mdp_graph`.

      Typical value: `"joint_3x3_gridworld_agent_A_uniform_random"`

  Here are the `parameters` you need to define for the `'multiagent_with_reward'` experiment type. You can find an example config file for the `'multiagent_with_reward'` experiment type under `configs/test/test_reward_correlation.yaml`.

    - `joint_mdp_graph`: The name of the file in the `mdps/` folder that corresponds to the joint MDP graph you want to use for your two agents. This name should not include the `.gv` extension.
    
      This should be a **joint multi-agent** MDP graph, not a single-agent graph. Best practice is to add the prefix `'joint_'` to all joint MDP graph filenames when [saving them](#save-an-mdp-graph), so it should be easy to tell which type of graph you're working with.

      Typical value: `"joint_3x3_gridworld"`

    - `seed_policy_graph_agent_A`: The name of the file in the `policies/` folder that corresponds to the [seed policy](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#A_1_Initial_optimal_policies_of_Agent_H) you want to assign to Agent A. This name should not include the `.gv` extension. POWERplay will train Agent H against this seed policy, and then train Agent A against the learned policies of Agent H. It will then use the multi-agent POWER formula to calculate [the POWERs of Agent H](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2_1_Multi_agent_POWER_for_Agent_H) and [the POWERs of Agent A](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#2_2_Multi_agent_POWER_for_Agent_A).

      This policy graph should be **compatible** with `joint_mdp_graph` above, in the sense that its sets of states and actions should be the same as those of `joint_mdp_graph`. Best practice is to prefix a policy filename with the _entire filename_ of the MDP it's associated with when [saving a policy](#save-a-policy-graph). So it should be easy to tell from the filenames whether `seed_policy_graph_agent_A` is compatible with `joint_mdp_graph`.

      Typical value: `"joint_3x3_gridworld_agent_A_uniform_random"`
    
    - `discount_rate_agent_A`: The discount rate for Agent A. Should be a number betwee 0 and 1. Typically this will be the same as `discount_rate` (which is the discount rate for Agent H in a multi-agent setting), but you might sometimes want to do experiments where the two agents have different discount rates.

      Typical value: `0.6`
    
    - `reward_correlation`: The correlation coefficient between the reward functions for Agent H and the reward functions for Agent A. If this is set to 1, then Agent A will always have exactly the same reward function as Agent H. If it's set to 0, then Agent A's reward function will always be independent of Agent H's reward function.

      If this is set to a number between 0 and 1, POWERplay uses [the interpolation scheme described here](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#3_4_Overcoming_instrumental_misalignment) to sample Agent A's reward function based on Agent H's reward function.

      It's also possible to set `reward_correlation` to a number between -1 and 0, if your reward function distribution is **symmetric**. Symmetric reward function distributions in `DISTRIBUTION_DICT` (`src/lib/utils/dist.py`) have a defined key called `'symmetric_interval'`, that defines the bounds over which the distribution is symmetric. But it's fairly rare in our experience investigate negative reward correlations, but qualitatively these correspond to [S-risk scenarios](https://longtermrisk.org/s-risks-talk-eag-boston-2017/) in our human-AI multi-agent setting.

      Typical value: `0`

You set the `parameters` above with different syntax depending on whether that parameter's value is **fixed**, or whether it **varies** over the course of an experiment's parameter sweep. If the value is **fixed**, then the syntax puts the actual value of the parameter under a `value` key in the `parameters` dictionary, like so:

```
discount_rate:
  value:
    0.1
```

If the value **varies** over the experiment, that means the parameter takes on multiple values over the course of a sweep. The way to specify this is to create a `values` key under which you list each value of the parameter, and a `names` key under which you list the names of the corresponding values in order to allow each run of your sweep to be **named** according to the parameter values it corresponds to. For example:

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

(Note that it's best to use "p" instead of "." for decimals, since these parameter names are going to be used in filenames.)

For examples of this, see the experiment config files under `configs/test/` and `configs/replication`.

### Visualizing experiment results

## Advanced usage

Here we'll list some useful functions outside the `base` module. We haven't fully documented these, but they can serve as starting points.

These modules are all imported by `main.py`, so you can access them easily.

### `graph` module: queries MDP and policy graphs

- `graph.is_graph_multiagent()`: Check if a policy or MDP graph is multi-agent or not.
- `graph.get_states_from_graph()`: Return the list of all the states in the input graph, in canonical order.
- `graph.get_actions_from_graph()`: Return the list of all the actions in the input graph, in canonical order. If the input graph is a joint MDP graph, this will return joint actions rather than single-agent actions.
- `graph.get_single_agent_actions_from_joint_mdp_graph()`: Return the lists _signle-agent_ actions that each agent (Agent H and Agent A) can take on a joint MDP graph.
- `graph.get_available_actions_from_graph_state()`: Given an input graph (MDP or policy) and a state in that input graph, return allowed actions from that state.
- `graph.get_available_states_and_probabilities_from_mdp_graph_state_and_action()`: Given an input MDP graph, a state, and an allowed action from that state, return the list of next-states from that action, and their probabilities.
- `graph.graph_to_joint_transition_tensor()`: Transforms a joint multi-agent MDP graph into a rank-4 tensor used to calculate POWERs. Tensor dimensions: (current states, Agent H actions, Agent A actions, next-states).
- `graph.graph_to_full_transition_tensor()`: Transforms a single-agent MDP graph into a rank-3 tensor used to calculate POWERs. Tensor dimensions: (current states, actions, next-states).
- `graph.graph_to_policy_tensor()`: Transforms a policy graph into a rank-2 tensor used to calculate POWERs. Tensor dimensions: (states, actions).
- `graph.any_graphs_to_full_transition_tensor()`: Combines graphs together to get a full transition tensor. If the input graph is a single-agent MDP, transforms it into a tensor. If the input graphs are a list of [joint multi-agent MDP graph, policy graph], combine them together to obtain the resulting single-agent MDP tensor.

### `mdp` module: creates and updates basic MDP graphs

- `mdp.add_state_action()`: Manually add a state-action-next-state subgraph to an MDP graph. The input `action_dict` is a dict of actions, next-states and their probabilities, and should look like this:

  ```
  {
      action_1: { state_1: prob_1 },
      action_2: { state_2: prob_2, state_3: prob_3 },
      action_3: { state_4: prob_4, state_5: prob_5 }
  }
  ```
- `mdp.update_state_action()`: Same as `mdp.add_state_action()`, except that you are updating an existing state instead of adding a new one.
- `mdp.remove_state_completely()`: Completely remove a state and its associated transition from an MDP graph.
- `mdp.generate_noised_gridworlds()`: Convenience function to generate a list of gridworlds with different stochastic noise levels. This can be useful if you want to sweep over noise levels in an experiment.

### `multi` module: creates and updates multi-agent graphs

- `multi.remove_states_with_overlapping_agents()`: Given a joint multi-agent MDP graph, delete all states at which the agents are at overlapping positions.
- `multi.create_joint_multiagent_graph()`: Given a single-agent graph (MDP or policy), transform that graph into a joint multi-agent graph.

### `policy` module: creates and updates policy graphs, manages policy rollouts

- `policy.update_state_actions()`: Manually update a state-action subgraph in a policy graph. The input `new_policy_actions` is a dict of actions and their probabilities, and should look like this:

  ```
  {
      action_1: prob_1,
      action_2: prob_2,
      action_3: prob_3
  }
  ```
- `policy.single_agent_to_multiagent_policy_graph()`: Given a single-agent policy graph, converts it to a joint multi-agent policy graph.
- `policy.sample_optimal_policy_data_from_run()`: Given the properties of a run (the output of `get.get_properties_from_run()`) and the index of a reward function sample you're interested in, return a dict that includes the optimal policy (or policies) of your agent(s) on that reward function sample. This is useful for troubleshooting or developing intuitions for specific cases.
- `policy.simulate_policy_rollout()`: Given an initial state, a policy, and an MDP graph, simulate a rollout of that policy starting from that state. The output is a list of states that define the rollout.
- `policy.policy_tensor_to_graph()`: Given a policy tensor (rank-2) and the MDP graph associated with that policy, reconstitute the original policy graph.

### `get` module: high-level functions for retrieving and preprocessing experimental results

- `get.get_properties_from_run()`: Given a sweep id (the unique number that prefixes the saved name of an experiment in `expts/`) and a run suffix (the unique identifier for each run of the sweep, the output of `get.get_sweep_run_suffixes_for_param()`), return all the data on that run of the sweep. This includes all the inputs to the sweep, the associated MDP and policy graphs, and outputs like the reward and policy samples generated in the experiment.
- `get.get_sweep_run_suffixes_for_param()`: Given a sweep id and the name of a parameter that varied over the sweep (named as it is in the [experiment config file](#experiment-config-files)), return the full list of run suffixes for that parameter. Useful for calling `get.get_properties_from_run()`.
- `get.get_sweep_type()`: Given a sweep id, return the experiment type. This will be either `'single_agent'`, `'multiagent_fixed_policy'`, or `'multiagent_with_reward'`. (Or will raise an exception.)
- `get.get_transition_graphs()`: Given the parameters of a run (i.e., the output of `get.get_sweep_run_results(sweep_id, run_suffix, results_type='inputs')`), return the MDP and policy graphs associated with a sweep. The flag `check_graph_compatibilities` runs a check to confirm that all the graphs are mutually compatible in terms of states and actions.
- `get.get_sweep_state_list()`: This is a convenience function that lets you quickly retrieve the list of states for a sweep. This function assumes that all runs in the sweep have the same set of states, even though the MDP itself may change over the sweep. This seems sensible, since if you're sweeping across runs you'll generally only be changing the transition probabilities of the MDP as opposed to the list of states itself.

### `viz` module: visualizations that get run automatically during experiments

- `viz.plot_sample_aggregations()`: Given a list of samples (can be reward samples or POWER samples, typically part of the output to `get.get_properties_from_run()`) and the MDP states associated with them, plot an aggregation of the samples. Supported aggregations are 1) `'mean'`, which plots the sample means and which is used for POWER calculations, and 2) `'var'`, which plots the sample variances and can be useful for debugging.
- `viz.plot_sample_distributions()`: Given a list of samples (can be reward samples or POWER samples, typically part of the output to `get.get_properties_from_run()`) and the MDP states associated with them, plot histograms showing the distributions of the samples on each state.
- `viz.plot_sample_correlations()`: Given a list of samples (can be reward samples or POWER samples, typically part of the output to `get.get_properties_from_run()`) and the MDP states associated with them, plot 2-D histograms showing the correlations between the samples at each state and the samples at every other state.

### `anim` module: creates & organizes animations from existing image files

- `anim.animate_from_filenames()`: Given a list of image filenames, create an animation of those filenames and output the result as a gif file.

### `learn` module: implementations of RL algorithms for value, policy, and POWER

- `learn.value_iteration()`: Given a reward function, discount rate, and MDP transition tensor, return a rank-1 tensor that corresponds to the optimal value function calculated via value iteration.
- `learn.policy_evaluation()`: Given a reward function, disount rate, MDP transition tensor, and policy tensor, return a rank-1 tensor that corresponds to the value function for the input policy tensor. This is from Sutton & Barto, Section 4.1 ("Iterative Policy Evaluation"). Note that in our setting, the reward `r` [depends only on the current state](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#fn3lmk7ctvj4z) `s`, not directly on the action `a`. This means at each iteration, the reward we actually capture always corresponds to the state we are on (i.e. `reward_function[state]`).
- `learn.find_optimal_policy()`: Given a reward function, discount rate, and MDP transition tensor, return a rank-2 tensor that corresponds to the optimal policy on that MDP. In our setting, the reward `r` [depends only on the current state](https://www.alignmentforum.org/posts/cemhavELfHFHRaA7Q/misalignment-by-default-in-multi-agent-systems#fn3lmk7ctvj4z) `s`, not directly on the action `a`. This means we can simplify the argmax expression for the policy (see the last line of the Value Iteration algorithm in Section 4.4 of Sutton & Barto) to eliminate the `r` term and the `gamma` factor. i.e., we can find the optimal policy from the optimal value function by simply taking `\argmax_a \sum_{s'} p(s' | s, a) V(s')`.

  **NOTE:** Whenever two action-values are equal, we intentionally **randomize** the policy over them to avoid systematically biasing our policies according to the canonical ordering of the states. This matters when we use a reward function distribution that's discrete rather than continuous (e.g., a Bernoulli distribution) because in this case there may be many states whose rewards are exactly identical. In cases like these, our choice of how to "break ties" between states with identical rewards can influence the POWER calculation significantly in the multi-agent setting. For example, we've observed that deterministic Agent H "tiebreaker" policies are very exploitable for Agent A, compared to stochastic Agent H "tiebreaker" policies.
- `learn.compute_power_values()`: Given a reward function, optimal value function, and discount rate, compute the POWER sample associated with that reward function. An [actual POWER calculation](https://www.alignmentforum.org/posts/pGvM95EfNXwBzjNCJ/instrumental-convergence-in-single-agent-systems#2_1_Definition) involves taking the expectation value (i.e., the mean) over all those reward samples.

### `runex` module: runs experiments

- `runex.run_one_experiment()`: Runs a full experiment given a set of transition graphs, discount rate, and reward function sampler. You normally run experiments via the `base.launch_experiment()` or `launch.launch_sweep()` APIs, but this function is sometimes useful in debugging runs.

### `dist` module: low-level functions for building & managing reward function distributions

- `dist.DISTRIBUTION_DICT`: A `dict` that contains a few pre-defined reward function distributions. The high-level key is what you use to access that distribution from the `reward_distribution` parameter of the [experiment config API](#experiment-config-files). It's possible to add new types of reward function distributions fairly easily: you can either use the [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) API, or define a distribution manually using a Python `lambda` function and the `misc.pdf_sampler_constructor()` function.
- `dist.config_to_pdf_constructor()`: Given a pdf `dict` of the form

  ```
  {
    'dist_name': <key for distribution in DISTRIBUTION_DICT>,
    'params': <params input to that distribution>
  }
  ```

  return a function that samples from that distribution.
- `dist.config_to_reward_distribution()`: Given a canonically ordered state list and a `reward_dist_config` of the form

  ```
    {
      'default_dist': <1d pdf config applied iid to all states>,
      'state_dists': { 
        <state label 1>: <1d pdf config applied to state 1 that overrides the default dist>
        <state label 2>: <1d pdf config applied to state 2 that overrides the default dist>
        ... etc.
      },
      'allow_all_equal_rewards': True,
      'states_with_identical_rewards': [
        [<state label 1, <state label 2], [<state label 3, <state label 4>]
      ]
    }
  ```

  return a `reward_sampler()` function you can call to sample sets of reward functions from the input distribution. For example, calling `reward_sampler(100)` would return a rank-2 tensor whose rows correspond to reward function samples, and whose columns correspond to states of the MDP in their canonical order.
- `dist.generate_correlated_reward_samples()`: Inputs: a single-agent reward distribution (`single_agent_reward_dist`, the output of `dist.config_to_reward_distribution()`); a list of actual reward function samples for Agent H (the output of `reward_sampler()`, where `reward_sampler()` is itself the output of `dist.config_to_reward_distribution()`); and a correlation coefficient between the reward functions of Agent H and Agent A. Output: a list of reward function samples for Agent A that both 1) satisfies the input correlation coefficient, and 2) corresponds to the marginal reward function distribution given by the input single-agent reward distribution `single_agent_reward_dist`.
