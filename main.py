import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
import pymoo.operators.sampling.rnd as pmrnd
from pymoo.termination import get_termination


class EuiclideanDistancePrivider:
    def __init__(self, positions):
        self.positions = positions
        
    def __getitem__(self, idx):
        i, j = idx
        pos1 = self.positions[i]
        pos2 = self.positions[j]

        return np.linalg.norm(pos1 - pos2)

    def __len__(self):
        return len(self.positions)



class UAVProblem(ElementwiseProblem):

    def __init__(self, Nv, Nt, Nm, succ_P, target_values, delta, max_distance, distances, eta):
        """
        Nv:       Number of UAVs
        Nt:       Number of targets
        Nm:       Number of tasks
        succ_P:   Probability of success of task on target k performed by UAV i (shape Nv x Nt)
        Value:    Value of target k
        """
        self.uav_count_Nv = Nv
        self.target_count_Nt = Nt
        self.task_count_Nm = Nm
        self.P = succ_P
        self.target_values = target_values
        self.delta = delta
        self.max_distance = max_distance
        self.distances = distances
        self.eta = eta
        
        super().__init__(
            n_var=Nv*Nt*Nm,
            n_obj=2,
            n_constr=Nm*Nt + Nv*Nt + Nm*Nv*Nt + Nv,
            xl=0,
            xu=1,
        )
        
    def _evaluate(self, x, out, *args, **kwargs):
        print("asdfasdfs --------------------------------- asdfasdfasdfasdf")
        # print(x)
        x = x.reshape((self.uav_count_Nv, self.target_count_Nt, self.task_count_Nm))
        # print(x)
        # print("1111 --------------------------------- 1111")

        for i in range(self.uav_count_Nv):
            pass
            # total_distance = 0

            # tasks = np.argwhere(x[i] == 1)
            # tasks2 = np.where(x[i] == 1)
            # tasks3 = x[i] == 1
            # print("Tasks")
            # print(tasks)
            # print(tasks2)
            # print(tasks3)

        
        # Objective 1: Function value
        # f1_target_value = 1 - np.sum(self.P * self.Value[:, None] * x) / self.delta
        print("P shape: ", self.P.shape)
        print("Value shape: ", self.target_values.shape)
        print("x shape: ", x.shape)
        # k and j in x matrix are reversed on purpose
        f1_target_value = 1 - np.einsum("ik,k,ikj->", self.P, self.target_values, x) / self.delta
        # f1_target_value = 1 - np.einsum("ik,k,ijk->j", self.P, self.Value, x) / self.delta
        # f1_target_value = 1 - np.einsum("k,ikj->", self.Value, x) / self.delta
        print("F1: ", f1_target_value)
        
        # Objective 2: Distance cost function
        total_distance = 0
        # for i in range(self.Nv):
        #     for j in range(self.Nt):
        #         total_distance += np.sum(self.Distance[i, j] * x[i, j])
        f2_distance_cost = total_distance / self.eta
        
        print("F1: ", f1_target_value)
        print("F2: ", f2_distance_cost)
        out["F"] = [f1_target_value, f2_distance_cost]

        
        # Constraints
        g = []
        
        # Constraint: Each task can be performed only once
        g.extend(np.sum(x, axis=0).reshape(-1) - 1)


        # Constraint: All tasks must be completed
        g.append(np.sum(x) - self.task_count_Nm * self.target_count_Nt)
        

        # Constraint: Each UAV can perform at most one task for each target
        for i in range(self.uav_count_Nv):
            for k in range(self.task_count_Nm):
                g.append(np.sum(x[i, :, k]) - 1)
        
        
        # # Flight distance constraint
        # for i in range(self.Nv):
        #     total_distance = 0
        #
        #     tasks = np.argwhere(x[i] == 1)
        #     tasks2 = np.where(x[i] == 1)
        #     tasks3 = x[i] == 1
        #     print("Tasks")
        #     print(tasks)
        #     print(tasks2)
        #     print(tasks3)
        #
        #     for n in range(self.N):
        #         total_distance += self.Distance[i, n, n+1]
        #     g.append(total_distance - self.Lmax[i])
        
        while len(g) < self.n_constr:
            g.append(0)

        print("G", g)
        out["G"] = g


# Example parameters
dron_count_Nv = 3
target_count_Nt = 3
tasks_count_Nm = 2
P = np.random.rand(dron_count_Nv, target_count_Nt)
target_values = np.random.rand(target_count_Nt)
delta = 1.0
l_max = np.random.rand(dron_count_Nv) * 100
points = np.random.rand(dron_count_Nv, target_count_Nt + 1, 2)
distance_provider = EuiclideanDistancePrivider(points)
eta = 1.0

# Create problem instance
problem = UAVProblem(dron_count_Nv, target_count_Nt, tasks_count_Nm, P, target_values, delta, l_max, distance_provider, eta)

# Set up algorithm
algorithm = NSGA2(
    pop_size=100,
    sampling=pmrnd.BinaryRandomSampling(), # type: ignore
)

# Set up termination criteria
termination = get_termination("n_gen", 20)

# Solve the problem
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True,
               eliminate_duplicates=True,
)

# Print results
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))


