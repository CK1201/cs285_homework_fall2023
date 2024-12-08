## Setup

See [installation.md](installation.md). It's worth going through this again since some dependencies have changed since homework 1. You also need to make sure to run `pip install -e .` in the hw2 folder.

## Running on Google Cloud
Starting with HW2, we will be providing some infrastructure to run experiments on Google Cloud compute. There are some very important caveats:

- **Do not leave your instance running.** The provided infrastructure tries to prevent this, but it will still be easy to accidentally leave your instance running and burn through all of your credits. You are responsible for making sure you use your credits wisely.
- **Only use this for big hyperparameter sweeps.** Definitely don't use Google Cloud for debugging; only launch a job once you are 100% sure your code works. Even then, single jobs will probably run faster on your local machine (yes, even if you don't have a GPU). The only reason to use Google Cloud is if you want to run multiple jobs in parallel.

For more instructions, see [google_cloud/README.md](google_cloud/README.md).

## Complete the code

There are TODOs in these files:

- `cs285/scripts/run_hw2.py`
- `cs285/agents/pg_agent.py`
- `cs285/networks/policies.py`
- `cs285/networks/critics.py`
- `cs285/infrastructure/utils.py`

See the [Assignment PDF](hw2.pdf) for more info.

## Running

```
python cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name cartpole --video_log_freq 10
python cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name cartpole_rtg -rtg --video_log_freq 10
python cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name cartpole_na -na --video_log_freq 10 
python cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name cartpole_na_rtg -na -rtg --video_log_freq 10
python cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 4000 --exp_name cartpole_lb --video_log_freq 10
python cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 4000 --exp_name cartpole_lb_rtg -rtg --video_log_freq 10
python cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 4000 --exp_name cartpole_lb_na -na --video_log_freq 10 
python cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 4000 --exp_name cartpole_lb_na_rtg -na -rtg --video_log_freq 10

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah --video_log_freq 10
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline --video_log_freq 10

python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda <λ> --exp_name lunar_lander_lambda_<λ>
```