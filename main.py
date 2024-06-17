import numpy as np
from typing import Optional
from numpy._typing import ArrayLike
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
import pymoo.operators.sampling.rnd as pmrnd
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.termination import get_termination
import fast_tsp as tsp
import time

DEBUG_LEVEL = 0

class EuclideanDistanceProvider:
    def __init__(self, positions1, positions2):
        self.positions1 = positions1
        self.positions2 = positions2
        
    def __getitem__(self, idx):
        i, j = idx
        if DEBUG_LEVEL >= 3:
            print("Getting distance between", i, j)
        pos1 = self.positions1[i]
        if DEBUG_LEVEL >= 3:
            print("Pos1", type(pos1), pos1)
        pos2 = self.positions2[j]
        if DEBUG_LEVEL >= 3:
            print("Pos2", type(pos2), pos2)

        return np.linalg.norm(pos1 - pos2)

    def __len__(self):
        return len(self.positions1) * len(self.positions2)


class UavsAndTargets:
    def __init__(self, uavs_positions, target_positions) -> None:
        self.uavs_positions = uavs_positions
        self.target_positions = target_positions

        self.uav_to_target_distances = EuclideanDistanceProvider(uavs_positions, target_positions)
        self.between_target_distances = EuclideanDistanceProvider(target_positions, target_positions)

    def get_euclidean_distance(self, uav_idx, target_idx):
        return self.uav_to_target_distances[uav_idx, target_idx]

    @property
    def uav_count(self):
        if type(self.uavs_positions) is list:
            return len(self.uavs_positions)

        return self.uavs_positions.shape[0]

    @property
    def target_count(self):
        if type(self.target_positions) is list:
            return len(self.target_positions)

        return self.target_positions.shape[0]


def find_visited_targets(x):
    Nv = x.shape[0]
    Nt = x.shape[1]
    Nm = x.shape[2]

    visited_targets_mask = np.any(x, axis=2)

    res = [np.argwhere(row).reshape(-1) for row in visited_targets_mask]

    return res

def find_path(uav: int, visited_targets: list[int], uavs_and_targets: UavsAndTargets):
    if len(visited_targets) == 0:
        return []
    target_distances = uavs_and_targets.between_target_distances[visited_targets, visited_targets]
    uav_to_target_distances = uavs_and_targets.uav_to_target_distances[uav, visited_targets]

    all_distances = np.zeros((len(visited_targets) + 1, len(visited_targets) + 1))

    # print("asdfsdfasdfasdfasfasdfasdfasdfasfasdfasdf===============================================")
    all_distances[0, 1:] = uav_to_target_distances
    all_distances[1:, 0] = uav_to_target_distances
    all_distances[1:, 1:] = target_distances
    # print(all_distances)

    dm = all_distances.max()

    all_distances = all_distances / dm
    all_distances = all_distances * 100000
    all_distances = all_distances.astype(int)

    # print(all_distances, type(all_distances), all_distances.shape, all_distances.dtype)
    # print(all_distances.tolist(), type(all_distances.tolist()), type(all_distances.tolist()[0]))

    path = tsp.find_tour(all_distances.tolist(), duration_seconds=5)

    path = [visited_targets[i-1] if i != 0 else 'uav' for i in path]


    return path



def distance_cost(visited_targets, uavs_and_targets: UavsAndTargets) -> ArrayLike | list[float]:
    global DEBUG_LEVEL
    distances = []

    for i in range(uavs_and_targets.uav_count):
        distance = 0
        if DEBUG_LEVEL >= 2:
            print("UAV: ", i)
            print(f"Visited targets for UAV {i}: ", visited_targets[i])
        path = find_path(i, visited_targets[i].tolist(), uavs_and_targets)
        if len(path) == 0:
            distances.append(0)
            continue
        path.append(path[0])
        if DEBUG_LEVEL >= 2:
            print("Path: ", path)
        for j in range(len(path)-1):
            val = path[j]
            if DEBUG_LEVEL >= 2:
                print(i, path, val)

            if val == 'uav':
                d = uavs_and_targets.uav_to_target_distances[i, path[j+1]]
            elif path[j+1] == 'uav':
                d = uavs_and_targets.uav_to_target_distances[i, val]
            else:
                d = uavs_and_targets.between_target_distances[val, path[j+1]]


            distance += d

        distances.append(distance)

    return distances



class UAVProblem(ElementwiseProblem):
    def __init__(self, uavs_and_targets: UavsAndTargets, Nm: int, succ_P, target_values, delta, max_distances, task_to_complete: Optional[int] = None):
        """
        Nv:             Number of UAVs
        Nt:             Number of targets
        Nm:             Number of tasks
        succ_P:         Probability of success of task on target k performed by UAV i (shape Nv x Nt)
        target_values:  Value of target k
        """
        self.uavs_and_targets = uavs_and_targets
        Nv = uavs_and_targets.uav_count
        Nt = uavs_and_targets.target_count

        self.task_to_complete = task_to_complete or Nm * Nt

        assert Nv == uavs_and_targets.uav_count
        assert Nv == succ_P.shape[0]
        assert Nt == target_values.shape[0]
        assert delta > 0

        self.task_count_Nm = Nm
        self.succ_prob = succ_P
        self.target_values = target_values
        self.delta = delta
        self.max_distance = max_distances
        
        super().__init__(
            n_var=Nv*Nt*Nm,
            n_obj=3,
            # n_obj=1,
            n_constr=Nm*Nt + Nv*Nt + 1 + Nv, # *2 is to force equality
            xl=0,
            xu=1,
        )

    @property
    def uav_count_Nv(self):
        return self.uavs_and_targets.uav_count

    @property
    def target_count_Nt(self):
        return self.uavs_and_targets.target_count

    @property
    def all_tasks_count(self):
        return self.target_count_Nt * self.task_count_Nm

    def format_x(self, x):
        return np.round(x).reshape((self.uav_count_Nv, self.target_count_Nt, self.task_count_Nm))
        
    def _evaluate(self, x, out, *args, **kwargs):
        global DEBUG_LEVEL
        constraints = []
        x = self.format_x(x)
        if DEBUG_LEVEL >= 1:
            print()
            print("----------------------------- NEW GENERATION -----------------------------")
            print()
        
            print("P shape: ", self.succ_prob.shape)
            print("Value shape: ", self.target_values.shape)
            print("x shape: ", x.shape)

            print(x)

        # Objective 1: Function value
        # k and j in x matrix are reversed on purpose (relative to original paper)
        f1_target_value = 1 - np.einsum("ik,k,ikj->", self.succ_prob, self.target_values, x) / self.delta
        
        visited_targets = find_visited_targets(x)
        if DEBUG_LEVEL >= 1:
            print("Visited targets: ", visited_targets)
        paths_lengths = distance_cost(visited_targets, self.uavs_and_targets)


        # Objective 2: Distance cost function
        f2_distance_cost = np.sum(paths_lengths)

        f3_tasks_completed = x.sum()
        
        out["F"] = [f1_target_value, f2_distance_cost, -f3_tasks_completed]
        # out["F"] = [f1_target_value + f2_distance_cost - f3_tasks_completed]
        if DEBUG_LEVEL >= 1:
            print("F", out["F"])
        

        # Constraint: Each task can be performed only once
        task_performed_once_constraints = np.sum(x, axis=0).reshape(-1) - 1
        constraints.extend(task_performed_once_constraints)
        assert len(task_performed_once_constraints) == self.task_count_Nm * self.target_count_Nt


        # Constraint: Enough tasks must be completed
        tasks_completed_constraint = self.task_to_complete - f3_tasks_completed
        tasks_completed_constraint = tasks_completed_constraint / self.all_tasks_count # normalization
        assert type(tasks_completed_constraint) is np.float64 or type(tasks_completed_constraint) is np.float32
        constraints.append(tasks_completed_constraint)
        

        # Constraint: Each UAV can perform at most one task for each target
        one_task_per_target_per_uav = []
        for i in range(self.uav_count_Nv):
            for j in range(self.target_count_Nt):
                one_task_per_target_per_uav.append(np.sum(x[i, j, :]) - 1)
        assert len(one_task_per_target_per_uav) == self.uav_count_Nv * self.target_count_Nt
        constraints.extend(one_task_per_target_per_uav)
        

        # Flight distance constraint
        distance_constrints = paths_lengths - self.max_distance
        assert len(distance_constrints) == self.uav_count_Nv
        constraints.extend(distance_constrints)
        
        out["G"] = constraints
        if DEBUG_LEVEL >= 1:
            print("G", out["G"])


class GAOutput(Output):

    def __init__(self):
        super().__init__()
        self.x_mean = Column("x_mean", width=13)
        self.x_std = Column("x_std", width=13)
        self.columns += [self.x_mean, self.x_std]

    def update(self, algorithm):
        super().update(algorithm)
        self.x_mean.set(np.mean(algorithm.pop.get("X")))
        self.x_std.set(np.std(algorithm.pop.get("X")))
        with open('x_mean.txt', 'a') as f:
            f.write(str(np.mean(algorithm.pop.get("X"))) + ',\n')
        with open('x_std.txt', 'a') as f:
            f.write(str(np.std(algorithm.pop.get("X"))) + ',\n')



# Example parameters
dron_count_Nv = 5
target_count_Nt = 12
tasks_count_Nm = 3
P = np.random.rand(dron_count_Nv, target_count_Nt)
target_values = np.random.rand(target_count_Nt)
delta = 1.0
l_max = np.array([300] * dron_count_Nv)

uav_positions = np.random.rand(dron_count_Nv, 2)
target_positions = np.random.rand(target_count_Nt, 2)
uav_and_targets = UavsAndTargets(uav_positions, target_positions)

# Create problem instance
problem = UAVProblem(uav_and_targets, tasks_count_Nm, P, target_values, delta, l_max, task_to_complete=target_count_Nt * tasks_count_Nm - 5)

# Set up algorithm
# algorithm = GA(
algorithm = NSGA2(
    pop_size=100,
    sampling=pmrnd.BinaryRandomSampling(), # type: ignore
)

# Set up termination criteria
termination = get_termination("n_gen", 500)

start = time.time()

# Solve the problem
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True,
               eliminate_duplicates=True,
               # output=GAOutput(),

)

end = time.time()

# Print results
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
print("Elapsed time: ", end - start)


