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

    def __init__(self, Nv, Nt, Nm, P, Value, delta, Lmax, Distance, eta):
        self.Nv = Nv
        self.Nt = Nt
        self.Nm = Nm
        self.P = P
        self.Value = Value
        self.delta = delta
        self.Lmax = Lmax
        self.Distance = Distance
        self.eta = eta
        self.N = len(Distance) - 1
        
        super().__init__(n_var=Nv*Nt*Nm,
                         n_obj=2,
                         n_constr=Nv*Nt*Nm + Nv + 1,
                         xl=0,
                         xu=1,
        )
        
    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        x = x.reshape((self.Nv, self.Nt, self.Nm))
        print("asdfasdfs --------------------------------- asdfasdfasdfasdf")
        # print(x)
        # print("1111 --------------------------------- 1111")

        for i in range(self.Nv):
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
        f1_target_value = 1 - np.einsum("ik,k,ijk->", self.P, self.Value, x) / self.delta
        
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
        g.append(np.sum(x) - self.Nm * self.Nt)
        

        # Constraint: Each UAV can perform at most one task for each target
        for i in range(self.Nv):
            for k in range(self.Nt):
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
Nv = 3
Nt = 2
Nm = 2
P = np.random.rand(Nv, Nm)
Value = np.random.rand(Nt)
delta = 1.0
Lmax = np.random.rand(Nv) * 100
points = np.random.rand(Nv, Nt + 1, 2)
distance_provider = EuiclideanDistancePrivider(points)
eta = 1.0

# Create problem instance
problem = UAVProblem(Nv, Nt, Nm, P, Value, delta, Lmax, distance_provider, eta)

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


