import numpy as np 
from casadi import* 


class MPC:
    def __init__(self) -> None:
        pass

    def GetOptimalSolution(self, Vmaxs : list, versor_distances : list, Jqs : list, q_dots : list, last_alpha : float):
        cost_function = 0
        constraints = SX.zeros(len(Vmaxs), 1)
        alphas = SX.sym("alphas", len(Vmaxs), 1)
        
        # setting optimization variable limits (i.e ai,min and ai,max for i E [1,...,m])
        lbx = 0
        ubx = 1

        # creating a single scalar constraint value for each frame => len(Vmaxs) == horizon lenghts!
        for i in range(0, len(Vmaxs)):
            if (i == 0):
                # saving maximum velocity towards human previous the scaling
                vmax_human = np.dot(np.dot(np.transpose(versor_distances[i]), Jqs[i]), q_dots[i])
            
            constraints[i] = Vmaxs[i] - np.dot(np.dot(np.dot(np.transpose(versor_distances[i]), Jqs[i]), q_dots[i]), alphas[i])

        # creating quadratic cost function: Jcost(alpha) = 1/2 * |alpha_norm - 1|^2
        # where alpha_norm = norm of each alphai calculate for each frame
        cost_function = 1/2 * (alphas - 1).T@(alphas - 1)

        # solve the quadratic problem
        qp = {"x": alphas, "f" : cost_function, "g" : constraints}
        options = {'sparse':True}
        solver = qpsol('solver', 'qpoases', qp, options)

        # calculate solution. It is important to specify the bounds, both for the optimization
        # variables and the constraint functions (in this case g() >= 0 !)
        solution = solver(lbx = lbx, ubx = ubx, lbg = 0)

        # get optimal values, for each frame, and return only the first one
        alpha_optimals = solution["x"].full()

        # scaling applied velocity towards human
        vmax_human = vmax_human * alpha_optimals

        return alpha_optimals[0], vmax_human

