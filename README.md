# Multi-agent active perception with prediction rewards

This repository is the reference implementation of the paper ["Multi-agent active perception with prediction rewards"](https://arxiv.org/abs/2010.11835).

![Conceptual overview](https://github.com/laurimi/multiagent-prediction-reward/blob/main/imgs/thumbnail.png?raw=true)

Some further information is available in [this blog post](https://laurimi.github.io/research/2020/10/23/neurips.html).

If you find the work useful, please cite it as:
Mikko Lauri and Frans A. Oliehoek. "Multi-agent active perception with prediction actions", in Advances in Neural Information Processing Systems (NeurIPS), 2020.

BiBTeX entry:
```
@inproceedings{lauri2020multiagent,
  author    = {Mikko Lauri and Frans A. Oliehoek}, 
  title     = {Multi-agent active perception with prediction actions},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year    = {2020}
}
```

## Requirements
The code consists of a C++ backend for solving Dec-POMDPs and a Python frontend that implements the APAS algorithm presented in the paper.
Follow the steps below to install necessary requirements and compile the planner.

### System libraries
Install the required system libraries on a Ubuntu system by:
```
sudo apt-get install libboost-all-dev libeigen3-dev
```
Additionally, you need a C++ compiler that supports C++17, and CMake version 3.0 or later.

### Compilation of the C++ backend
You can compile the C++ backend by executing:
```
cd solver && mkdir build && cd build
cmake ..
make
```

Note: this will download and compile [MADP toolbox](https://github.com/MADPToolbox/MADP) version 0.4.1 which usually takes quite a long time.
If you already have MADP installed, you can save a lot of time by specifying where to find it: `cmake .. -DMADPPATH=/path/to/your/madp/installation`.

### Install Python requirements
Use Python3.
Only `numpy` is required.
You probably have it, or you can run:
```
pip install -r requirements.txt
```

## Example run
You can solve the MAV domain with horizon 5 using the experimental settings from the paper by running:
```
python apas.py --horizon 5 `pwd`/problems/mav.dpomdp --verbose
```
We toggled verbose output to get some printouts in the terminal.
Results will be stored in the subfolder `results`. There you will find the following contents:
- `apas_policy.out` indicates where to find the best individual policies found by APAS for each agent
- `apas_value.npy` a file that can be loaded using `np.load` containng the value of the best policy found by APAS
- `beliefs_XYZ.txt` text files containing on each row a belief state used as linearization point for the final reward at iteration `XYZ` of APAS
- `policy_values.npy` loadable numpy file with the value of the policy foudn at each iteration of APAS
- Subfolders `pgi_XY` containing the best individual policies and all individual policies considered by policy graph improvement for each agent at iteration `XY` of APAS. The files are in `.dot` format and can be visualized using `xdot`.

All values stored are exact.
The approximation by a piecewise linear function is *not* used when evaluating the policies, it is only used when planning.

## Results
The archives linked below contain the raw data corresponding to the results presented in the paper and supplementary material.
The format is similar to that described above.
- [APAS (*730 MB download*)](https://drive.google.com/file/d/1RtzyAf_7iPBloEkzCUXlcEUH6X7ITMJw/view?usp=sharing)
- [APAS without adaptation phase (*963 MB download*)](https://drive.google.com/file/d/1WzYZCcMwBdKsMPK1fMA48h6YEaUzUD8a/view?usp=sharing)

## Solving your own problems or using different final rewards
The software uses the parser from the MADP toolbox to read problems formatted as `.dpomdp` files.
You can specify your own problems in this format.
See [this example problem](https://github.com/MADPToolbox/MADP/blob/master/problems/dectiger.dpomdp) for a description of the format.

However, note that the `.dpomdp` format does not allow specifying rewards that are not linear in the belief state (i.e., functions of the hidden state and actions).
The planner software implicitly assumes you wish to solve a Dec-rhoPOMDP with negative entropy as the final reward.
If you want to use a different final reward, modify `DecPOMDPConversions.hpp`.
You will need to add functionality for getting the linearizing hyperplanes of your (convex and bounded) final reward function; see `LinearizedNegEntropy.hpp` for an example.

## Technical details
The conversion from Definition 3 in the paper is implemented in `DecPOMDPConversions.hpp`.
The main part of the Dec-POMDP solver is implemented in `BackwardPass.hpp`.
We use particle-based PGI, however modified with UCB1 applied to optimize node configurations.

## Contributing
Pull requests are welcome, although there are no plans for active further development as of now.

Licensed under the MIT license - see [LICENSE](LICENSE) for details.
