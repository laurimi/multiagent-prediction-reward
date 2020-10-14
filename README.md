# Multi-agent active perception with prediction rewards

This repository is the reference implementation of [Multi-agent active perception with prediction rewards](https://arxiv.org/abs/2030.12345).

If you find the work useful, please cite it as:
Mikko Lauri and Frans Oliehoek. "Multi-agent active perception with prediction actions", in Advances in Neural Information Processing Systems (NeurIPS), 2020.

BiBTeX entry:
```
@inproceedings{lauri2020multiagent,
  author    = {Mikko Lauri and Frans Oliehoek}, 
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
sudo apt-get install libboost-dev libeigen3-dev
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

## Results
You can download the outputs as indicated above that contain the raw data corresponding to the results presented in the paper and supplementary material.
The links are provided below.
- [APAS *730 MB*](https://drive.google.com/file/d/1RtzyAf_7iPBloEkzCUXlcEUH6X7ITMJw/view?usp=sharing)
- [APAS without adaptation phase *963 MB*](https://drive.google.com/file/d/1WzYZCcMwBdKsMPK1fMA48h6YEaUzUD8a/view?usp=sharing)

## Contributing
Licensed under the MIT license - see [LICENSE](LICENSE) for details.
