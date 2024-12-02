import numpy as np  
import openmdao.api as om



class ComputeAcceleration(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')
        self.options.declare('g', default=9.806, desc='gravitational acceleration')


    def setup(self):

        # Inputs    
        self.add_input('drag', val= np.ones(self.options['n']), desc='drag force', units='N')
        self.add_input('total_thrust_gen', val= np.ones(self.options['n']), desc='thrust force', units='N')
        self.add_input('mass', val=1, desc='mass', units='kg')
        self.add_input('gamma', val= np.ones(self.options['n']), desc='flight path angle', units='rad')

        # Outputs
        self.add_output('acc', val= np.zeros(self.options['n']), desc='acceleration in longitudinal direction of body-fixed frame', units='m/s**2')

    def setup_partials(self):
        self.declare_partials('acc', 'drag')    
        self.declare_partials('acc', 'total_thrust_gen')
        self.declare_partials('acc', 'mass')
        self.declare_partials('acc', 'gamma')

    def compute(self, inputs, outputs):

        # Unpack inputs
        Drag = inputs['drag']
        Thrust = inputs['total_thrust_gen']
        mass = inputs['mass']
        gamma = inputs['gamma']

        # Unpack constants
        g = self.options['g']

        # Accelerations in the body-fixed frame
        acc  = (Thrust - Drag - mass * g * np.sin(gamma)) / mass
    
        # Pack outputs
        outputs['acc'] = acc

    def compute_partials(self, inputs, J):

        # Unpack inputs
        Drag = inputs['drag']
        Thrust = inputs['total_thrust_gen']
        mass = inputs['mass']
        gamma = inputs['gamma']

        # Unpack constants
        g = self.options['g']

        # Compute partials
        J['acc', 'drag'] = -1 / mass
        J['acc', 'total_thrust_gen'] = 1 / mass
        J['acc', 'mass'] = -(Thrust - Drag  - g * np.sin(gamma)) / mass**2
        J['acc', 'gamma'] = -mass * g * np.cos(gamma) / mass


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('total_thrust_gen', 1000, units='N')
    ivc.add_output('lift', 1000, units='N')
    ivc.add_output('drag', 100, units='N')
    ivc.add_output('mass', 8600, units='kg')
    ivc.add_output('gamma', 0, units='rad')


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeAcceleration', ComputeAcceleration(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('acc = ', p['ComputeAcceleration.acc'])
