import numpy as np  
import openmdao.api as om



class ComputeVelocities(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')
        self.options.declare('g', default=9.806, desc='gravitational acceleration')


    def setup(self):

        # Inputs    
        self.add_input('lift', val= np.ones(self.options['n']), desc='lift force', units='N')
        self.add_input('drag', val= np.ones(self.options['n']), desc='drag force', units='N')
        self.add_input('total_thrust_gen', val= np.ones(self.options['n']), desc='thrust force', units='N')
        self.add_input('mass', val=1, desc='mass', units='kg')
        self.add_input('gamma', val= np.ones(self.options['n']), desc='flight path angle', units='rad')
        self.add_input('dt', val= np.ones(self.options['n']), desc='time step', units='s')
        self.add_input('v0', val=0, desc='initial velocity', units='m/s')

        # Outputs
        self.add_output('acc', val= np.zeros(self.options['n']), desc='acceleration in longitudinal direction of body-fixed frame', units='m/s**2')
        self.add_output('utas', val= np.ones(self.options['n']), desc='true airspeed', units='m/s', lower=1e-3)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        Lift = inputs['lift']
        Drag = inputs['drag']
        Thrust = inputs['total_thrust_gen']
        mass = inputs['mass']
        gamma = inputs['gamma']
        dt = inputs['dt']
        v0 = inputs['v0']

        # Unpack constants
        g = self.options['g']

        # Accelerations in the body-fixed frame
        acc  = (Thrust - Drag - mass * g * np.sin(gamma)) / mass

        # Compute velocity in the body-fixed frame
        utas = np.cumsum(acc * dt) + v0

    
        # Pack outputs
        outputs['acc'] = acc
        outputs['utas'] = utas


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
    ivc.add_output('dt', 0.1, units='s')
    ivc.add_output('v0', 0, units='m/s')


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeVelocity', ComputeVelocities(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('acc = ', p['ComputeVelocity.acc'])
    print('utas = ', p['ComputeVelocity.utas'])
