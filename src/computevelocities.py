import numpy as np  
import openmdao.api as om



class ComputeVelocities(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')


    def setup(self):

        # Inputs    
        self.add_input('dt', val= np.ones(self.options['n']), desc='time step', units='s')
        self.add_input('v0', val=0, desc='initial velocity in the x-direction of the body fixed axis', units='m/s')
        self.add_input('acc', val= np.zeros(self.options['n']), desc='acceleration in longitudinal direction of body-fixed frame', units='m/s**2')

        # Outputs
        self.add_output('vel', val= np.ones(self.options['n']), desc='true airspeed', units='m/s', lower=1e-3)

    def setup_partials(self):
        self.declare_partials('vel', 'acc')
        self.declare_partials('vel', 'v0')
        self.declare_partials('vel', 'dt')

    def compute(self, inputs, outputs):

        # Unpack inputs
        acc = inputs['acc']
        dt = inputs['dt']
        v0 = inputs['v0']

        # Compute velocity in the body-fixed frame
        vel = np.cumsum(acc * dt) + v0

    
        # Pack outputs
        outputs['vel'] = vel

    def compute_partials(self, inputs, J):

        # Unpack inputs
        dt = inputs['dt']
        acc = inputs['acc']

        n = self.options['n']

        # Compute partials
        J['vel', 'acc'] = np.eye(n) * dt
        J['vel', 'dt'] = np.eye(n) * acc
        J['vel', 'v0'] = 1



if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    n = 10

    ivc = om.IndepVarComp()

    ivc.add_output('dt', 0.1 * np.ones(n), units='s')
    ivc.add_output('acc', np.ones(n)*0.1, units='m/s**2')
    ivc.add_output('v0', 0, units='m/s')


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeVelocity', ComputeVelocities(n=n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('vel = ', p['ComputeVelocity.vel'])
