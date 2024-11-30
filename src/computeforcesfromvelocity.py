import numpy as np  
import openmdao.api as om



class ComputeVelocityFromForces(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')


    def setup(self):

        # Inputs    
        self.add_input('Lift', val= np.ones(self.options['n']), desc='lift force', units='N')
        self.add_input('Drag', val= np.ones(self.options['n']), desc='drag force', units='N')
        self.add_input('Thrust', val= np.ones(self.options['n']), desc='thrust force', units='N')
        self.add_input('mass', val= np.ones(self.options['n']), desc='mass', units='kg')
        self.add_input('gamma', val= np.ones(self.options['n']), desc='flight path angle', units='rad')
        self.add_input('dt', val= np.ones(self.options['n']), desc='time step', units='s')
        self.add_input('u0', val=0, desc='initial velocity in x-axis of body-fixed frame', units='m/s')
        self.add_input('w0', val=0, desc='initial velocity in z-axis of body-fixed frame', units='m/s')

        # Outputs
        self.add_output('ax', val=0, desc='acceleration in x', units='m/s**2')
        self.add_output('az', val=0, desc='acceleration in z', units='m/s**2')
        self.add_output('vtas', val=0, desc='true airspeed', units='m/s')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        Lift = inputs['Lift']
        Drag = inputs['Drag']
        Thrust = inputs['Thrust']
        mass = inputs['mass']
        gamma = inputs['gamma']
        dt = inputs['dt']
        u0 = inputs['u0']
        w0 = inputs['w0']

        # Unpack constants
        g = self.options['g']

        

        # Accelerations in the body-fixed frame
        axb = (Thrust - Drag - mass * g * np.sin(gamma)) / mass
        azb = (Lift - mass * g * np.cos(gamma)) / mass

        # Compute velocity in the body-fixed frame
        u = np.cumsum(axb * dt) + u0
        w = np.cumsum(azb * dt) + w0

        vtas = np.sqrt(u**2 + w**2)


        # Pack outputs
        outputs['ax'] = axb
        outputs['az'] = azb
        outputs['vtas'] = vtas


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('Thrust', 1000, units='N')
    ivc.add_output('Lift', 1000, units='N')
    ivc.add_output('Drag', 100, units='N')
    ivc.add_output('mass', 8600, units='kg')
    ivc.add_output('gamma', 0, units='rad')
    ivc.add_output('dt', 0.1, units='s')
    ivc.add_output('u0', 0, units='m/s')
    ivc.add_output('w0', 0, units='m/s')

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeVelocity', ComputeVelocity(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('ax = ', p['ComputeVelocity.ax'])
    print('az = ', p['ComputeVelocity.az'])
    print('vtas = ', p['ComputeVelocity.vtas'])
