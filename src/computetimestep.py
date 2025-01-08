import numpy as np  
import openmdao.api as om



class ComputeTimeStep(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')        


    def setup(self):

        # Inputs 
        self.add_input('t0', val= 0, desc='initial time', units='s')
        self.add_input('t1', val= 10 * 60, desc='end time', units='s')


        # Outputs
        self.add_output('dt', val= np.ones(self.options['n']), desc='time step', units='s', lower=1e-3)

    def setup_partials(self):
        self.declare_partials('dt', 't0')
        self.declare_partials('dt', 't1')

    def compute(self, inputs, outputs):

        # Unpack inputs
        t0 = inputs['t0']
        t1 = inputs['t1']

        # Unpack options
        n = self.options['n']

        t = np.linspace(t0, t1, n).flatten()
        dt = np.diff(t)
        dt = np.append(0, dt)

        if np.any(dt < 0):
            print('Negative time steps detected')


        # Pack outputs
        outputs['dt'] = dt

    def compute_partials(self, inputs, J):

        J['dt', 't0'] = -1
        J['dt', 't1'] = 1


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    n = 100

    ivc = om.IndepVarComp()
    ivc.add_output('t0', 0, units='s')
    ivc.add_output('t1', 10*60, units='s')

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeTimeStep', ComputeTimeStep(n = n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('dt = ', p['ComputeTimeStep.dt'])

