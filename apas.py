import numpy as np
import os
import subprocess
import argparse
from random import randrange
from pathlib import Path
from collections import defaultdict

def read_number_of_agents(dpomdp_file):
	"""Returns the number of agents in a Dec-POMDP problem

    Keyword arguments:
    dpomdp_file -- path to problem file in the .dpomdp format
    """
	with open(dpomdp_file) as file:
		for line in file:
			if line.startswith('agents:'):
				return int(line.split(':')[1])
	raise ValueError('Could not determine number of agents from file: {:s}'.format(dpomdp_file))

def read_number_of_states(dpomdp_file):
	"""Returns the number of states in a Dec-POMDP problem

    Keyword arguments:
    dpomdp_file -- path to problem file in the .dpomdp format
    """
	with open(dpomdp_file) as file:
		for line in file:
			if line.startswith('states:'):
				return int(line.split(':')[1])
	raise ValueError('Could not determine number of states from file: {:s}'.format(dpomdp_file))

class JointPolicyEvaluator:
	"""Evaluates values of a joint policy in a Dec-rhoPOMDP"""
	def __init__(self, dpomdp_file, horizon, evaluation_executable):
		self.num_agents = read_number_of_agents(dpomdp_file)
		self.base_executable = [evaluation_executable, '-l', '{:d}'.format(horizon), 
							 '-e', dpomdp_file, '--files']

	def __call__(self, individual_policies):
		"""Returns the value of a joint policy in a Dec-rhoPOMDP problem.

		Keyword arguments:
		individual_policies -- list of individual policy files for agents 0, 1, ..., (n-1)
		"""
		if len(individual_policies) != self.num_agents:
			raise ValueError("Number of agents {:d} does not match the number \
					of input individual policies {:d}".format(self.num_agents, len(individual_policies)))

		executable = self.base_executable + [p for p in individual_policies]
		sp = subprocess.check_output(executable)
		return self.parse_value(sp)

	def parse_value(self, output):
		"""Returns the value of a joint policy parsed from evaluator output

		Keyword arguments:
		output -- output from subprocess that performs evaluation
		"""
		for line in output.decode('ascii').split('\n'):
			if line.startswith('value'):
				return float(line.strip('\n').split('= ')[1])
		raise ValueError("Could not parse value from output")

class JointPolicyBeliefSampler:
	"""Samples belief states at end of executing a joint policy by rolling out trajectories"""
	def __init__(self, dpomdp_file, horizon, num_rollouts, rollout_executable):
		"""Updates the individual policies to be used

		Keyword arguments:
		dpomdp_file -- path to problem file in the .dpomdp format
		horizon -- planning horizon in Dec-rhoPOMDP
		num_rollouts -- number of trajectories to roll out
		rollout_executable -- executable for rolling out trajectories and collecting state statistics
		"""
		self.rollout_executable = rollout_executable
		self.dpomdp_file = dpomdp_file
		self.num_agents = read_number_of_agents(self.dpomdp_file)
		self.num_states = read_number_of_states(self.dpomdp_file)
		self.horizon = horizon
		self.num_rollouts = num_rollouts
		self.individual_policies = None
		self.probabilities = None
		self.beliefs = None

	def set_individual_policies(self, individual_policies):
		"""Updates the individual policies to be used

		Keyword arguments:
		individual_policies -- list of individual policy files for agents 0, 1, ..., (n-1)
		"""
		if len(individual_policies) != self.num_agents:
			raise ValueError("Number of agents {:d} does not match the number \
					of input individual policies {:d}".format(self.num_agents, len(individual_policies)))
		self.individual_policies = individual_policies
		self.rollout()

	def rollout(self):
		"""Run rollouts to generate new set of end beliefs and estimated probabilities"""
		seed = randrange(9999999)
		executable = [self.rollout_executable, '-l', '{:d}'.format(self.horizon), 
					'-n', '{:d}'.format(self.num_rollouts), self.dpomdp_file] \
					+ ['-g', '{:d}'.format(seed)] + ['--files'] \
					+ [p for p in self.individual_policies]
		sp = subprocess.check_output(executable)
		end_states = self.parse_end_states(sp)

		N = len(end_states)
		self.beliefs = np.zeros((N, self.num_states))
		self.probabilities = np.zeros((N,))

		for i, (k, states) in enumerate(end_states.items()):
			num_samples = len(states)
			self.probabilities[i] = float(num_samples) / float(self.num_rollouts)
			sample_contribution = 1.0 / float(num_samples)
			for s in states:
				self.beliefs[i, s] += sample_contribution


	def parse_end_states(self, output):
		"""parse the output from the rollout executable"""
		end_states = defaultdict(list)
		for line in output.decode('ascii').split('\n'):
			if line.startswith('time ='):
				execstate, s_str = line.split('-- ')
				end_states[execstate].append(int(s_str))
		return end_states

	def sample(self, N):
		"""Draw N samples of end beliefs"""
		num_beliefs = self.beliefs.shape[0]
		if N <= num_beliefs:
			# sample without replacement from the set of end beliefs generated via rollouts
			idx = np.random.choice(num_beliefs, size=N, replace=False, p=self.probabilities)
			return self.beliefs[idx,:]
		else:
			# return all rollout beliefs and sample more random beliefs so that we have enough
			us = UniformBeliefSampler(self.num_states)
			return np.vstack(self.beliefs, us.sample(N - num_beliefs) )

class UniformBeliefSampler:
	"""Samples belief states uniformly at random"""
	def __init__(self, dim):
		"""
		Keyword arguments:
		dim -- dimension of the belief to sample
		"""
		self.dim = dim

	def set_individual_policies(self, individual_policies):
		"""Do nothing, uniform sampler does not use policy"""
		pass

	def sample(self, N):
		"""
		Return N belief states sampled uniformly at random

		Implements the Kraemer algorithm as explained in
			Smith & Tromble (2004): "Sampling Uniformly from the Unit Simplex"
			https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
		"""
		return np.vstack([self.sample_single() for n in range(N)])

	def sample_single(self):
		x = np.random.randint(0, np.iinfo(np.int64).max, size=self.dim+1)
		x.sort()
		x[0] = 0
		x[-1] = np.iinfo(np.int64).max
		return np.diff(x).astype(np.float) / float(np.iinfo(np.int64).max)

class DecRhoPOMDPSolver:
	"""Frontend for PGI Dec-POMDP solver"""
	def __init__(self, solver_executable, width, n_rollouts, n_heur, n_improvements, p_random):
		"""
		Keyword arguments:
		solver_executable -- PGI solver executable
		width -- width of policy graphs for each agent
		n_rollouts -- number of rollouts
		n_heur -- number of heuristic samples
		n_improvements -- number of improvement iterations
		p_random -- policy graph node randomization probability to escape local minima
		"""
		self.solver_executable = solver_executable
		self.width = width
		self.n_rollouts = n_rollouts
		self.n_heur = n_heur
		self.n_improvements = n_improvements
		self.p_random = p_random

	def solve(self, dpomdp_file, horizon, beliefs_file, outputpath):
		"""Returns a list of filenames for individual policies

		Keyword arguments:
		dpomdp_file -- path to problem file in the .dpomdp format
		horizon - planning horizon in Dec-rhoPOMDP
		beliefs_file -- file containing belief states at which to linearize final reward function
		outputpath -- path where to write outputs
		"""
		num_agents = read_number_of_agents(dpomdp_file)
		cmd = [self.solver_executable] + ['-r', '{:d}'.format(self.n_rollouts), 
					'-n', '{:d}'.format(self.n_heur),
					'-p', '{:4f}'.format(self.p_random),
					'-i', '{:d}'.format(self.n_improvements), '-l', '{:d}'.format(horizon+1), 
					'-b', os.path.realpath(beliefs_file), dpomdp_file, '--width'] \
					+ ['{:d}'.format(self.width) for i in range(horizon+1)]

		Path(outputpath).mkdir(parents=True, exist_ok=True)
		sp = subprocess.check_output(cmd, cwd=outputpath)

		return [os.path.join(outputpath, 'agent_{:d}_best_policy.dot').format(k) for k in range(num_agents)]

def apas(dpomdp_file, N, K, horizon, solver, initial_belief_sampler, 
			end_belief_sampler, policy_evaluator, outputpath, verbose):
	"""Solves a Dec-rhoPOMDP problem by applying the APAS algorithm, writes results to disk

    Keyword arguments:
    dpomdp_file -- path to problem file in the .dpomdp format
    N -- number of iterations to run APAS (no. policy optimization + adaptation phases)
    K -- number of individual prediction actions per agent, or number of tangent hyperplanes
    horizon -- planning horizon in Dec-rhoPOMDP
    solver -- class implementing DecRhoPOMDPSolver
    initial_belief_sampler -- class for sampling belief states to use as initial linearization points
	end_belief_sampler -- class for sampling linearization belief states conditional on best policy
	policy_evaluator -- class implementing JointPolicyEvaluator
	outputpath -- path where results of APAS will be written
	verbose -- boolean switch for verbose output
    """
	if verbose:
		def verboseprint(*args, **kwargs):
			print(*args, **kwargs)
	else:   
		verboseprint = lambda *a, **k: None

	Path(outputpath).mkdir(parents=True, exist_ok=True)
	verboseprint('[{:d}/{:d}]\tOutput path: {:s}'.format(0, N, outputpath))

	# store overall best policy and value
	best_individual_policies, best_value = "", -9999999999.0

	# sample the initial beliefs to use as linearization points
	beliefs = initial_belief_sampler.sample(K)

	# store policy values per iteration
	policy_values = []
	for apas_iteration in range(N):
		# Save the linearization belief states used in this iteration
		bout = os.path.join(outputpath, 'beliefs_{:03d}.txt'.format(apas_iteration))
		np.savetxt(bout, beliefs, delimiter=',')

		# Policy optimization phase: solve Dec-POMDP and evaluate joint policy
		cwd = os.path.realpath(os.path.join(outputpath, 'pgi_{:02d}'.format(apas_iteration)))
		individual_policies = solver.solve(dpomdp_file, horizon, bout, cwd)
		value = policy_evaluator(individual_policies)
		if value > best_value:
			best_value, best_individual_policies = value, individual_policies

		policy_values.append(value)
		verboseprint('[{:d}/{:d}]\tPGI best so far: {:4f}'.format(apas_iteration, N, best_value))

		# Adaptation phase: sample new belief states to use as linearization points on next iteration
		end_belief_sampler.set_individual_policies(best_individual_policies)
		beliefs = end_belief_sampler.sample(K)
		verboseprint('[{:d}/{:d}]\tSelected new linearization points'.format(apas_iteration, N))

	verboseprint('[{:d}/{:d}]\tBest policy value: {:4f}'.format(N, N, best_value))
	verboseprint('[{:d}/{:d}]\tBest policy:'.format(N, N))
	verboseprint(''.join(best_individual_policies))

	vout = os.path.join(outputpath, 'policy_values.npy')
	np.save(vout, np.array(policy_values))

	best_out = os.path.join(outputpath, 'apas_value.npy')
	np.save(best_out, best_value)
	best_pol_out = os.path.join(outputpath, 'apas_policy.out')
	with open(best_pol_out, 'w') as b:
		b.writelines(best_individual_policies)

def parse_command_line():
	script_path = os.path.dirname(os.path.abspath(__file__))

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('problem', help='problem to solve as .dpomdp file')

	# Options for the algorithm in general
	utils_g = parser.add_argument_group(title='General options')
	utils_g.add_argument('--outputpath', type=str, 
		default=os.path.join(script_path, "results"), 
		help='output path for results')
	utils_g.add_argument('--eval', type=str, 
		default=os.path.join(script_path, "solver/build/value_fsc_policy"), 
		help='executable for evaluating policies')
	utils_g.add_argument('--endsampler', type=str, 
		default=os.path.join(script_path, "solver/build/fsc_policy_endstates"), 
		help='executable for sampling policy end states')
	utils_g.add_argument('--verbose', action='store_true', 
		help='toggle verbose output')
	
	# These are the options for the main APAS algorithm
	apas_g = parser.add_argument_group(title='APAS options')
	apas_g.add_argument('--horizon', type=int, default=2, 
		help='horizon')
	apas_g.add_argument('--niter', type=int, default=10, 
		help='number of iterations')
	apas_g.add_argument('--num_prediction_action',  type=int, 
		default=2, help='number of prediction actions')
	apas_g.add_argument('--num_rollouts', type=int, default=1000, 
		help='number of rollouts for adaptation phase')
	apas_g.add_argument('--nonadaptive', action='store_true', 
		help='toggle to disable adaptation phase of APAS')

	# These are the options for the PGI algorithm
	# PGI is called at each iteration of APAS to solve the standard Dec-POMDP
	pgi_g = parser.add_argument_group(title='PGI options')
	pgi_g.add_argument('--pgisolver', type=str, 
		default=os.path.join(script_path,"solver/build/improve_fsc_preds"), 
		help='PGI executable')
	pgi_g.add_argument('--pgi-rollouts', type=int, default=2000, 
		help='number of rollouts')
	pgi_g.add_argument('--pgi-heur-samples', type=int, default=50, 
		help='number of heuristic samples')
	pgi_g.add_argument('--pgi-num-improvements', type=int, 
		default=20, help='number of improvement iterations')
	pgi_g.add_argument('--width', type=int, default=2, 
		help='policy graph width')
	pgi_g.add_argument('--p-random', type=float, default=0.1, 
		help='node randomization probability')

	return parser

if __name__ == '__main__':
	args = parse_command_line().parse_args()

	pgi = DecRhoPOMDPSolver(args.pgisolver, args.width, 
		args.pgi_rollouts, args.pgi_heur_samples, 
		args.pgi_num_improvements, args.p_random)

	num_states = read_number_of_states(args.problem)
	belief_state_initializer = UniformBeliefSampler(num_states)

	if args.nonadaptive:
		# end belief sampler is uniform sampler in nonadaptive mode
		end_belief_sampler = belief_state_initializer
	else:
		end_belief_sampler = JointPolicyBeliefSampler(args.problem, 
			args.horizon, args.num_rollouts, args.endsampler)

	policy_evaluator = JointPolicyEvaluator(args.problem, 
		args.horizon, args.eval)

	pgi_solver = [args.pgisolver, '-r', '{:d}'.format(args.pgi_rollouts), 
					'-n', '{:d}'.format(args.pgi_heur_samples),
					'-p', '{:4f}'.format(args.p_random),
					'-i', '{:d}'.format(args.pgi_num_improvements)]

	apas(args.problem, args.niter, args.num_prediction_action, 
		args.horizon, pgi, belief_state_initializer, 
		end_belief_sampler, policy_evaluator, args.outputpath, 
		args.verbose)