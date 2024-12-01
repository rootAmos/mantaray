import numpy as np  
import openmdao.api as om



class ComputeTrajectories(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')        


    def setup(self):

        # Inputs    
        self.add_input('vx', val= np.ones(self.options['n']) * 0, desc='flight path angle', units='m/s')
        self.add_input('vz', val= np.ones(self.options['n']) * 0, desc='true airspeed', units='m/s')
        self.add_input('x0', val=  0, desc='initial distance', units='m')
        self.add_input('z0', val=  0, desc='initial altitude', units='m')
        self.add_input('t0', val= 0, desc='initial time', units='s')
        self.add_input('t1', val= 0, desc='end time', units='s')


        # Outputs
        self.add_output('x', val= np.ones(self.options['n']), desc='trajectory in the x-axis', units='m')
        self.add_output('dt', val=np.ones(self.options['n']), desc='time step', units='s')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        vx = inputs['vx']
        vz = inputs['vz']
        x0 = inputs['x0']
        z0 = inputs['z0']
        t0 = inputs['t0']
        t1 = inputs['t1']

        # Unpack options
        n = self.options['n']

        t = np.linspace(t0, t1, n).flatten()
        dt = np.diff(t)
        dt = np.append(0, dt)

        # Compute distance, time, and altitude
        x = np.cumsum(vx * dt) + x0

        # Pack outputs
        outputs['x'] = x
        outputs['dt'] = dt

if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    n = 100

    ivc = om.IndepVarComp()
    ivc.add_output('vx', 100 * np.ones(n), units='m/s')
    ivc.add_output('vz', 5 *np.ones(n), units='m/s')
    ivc.add_output('t0', 0, units='s')
    ivc.add_output('t1', 10*60, units='s')

    ivc.add_output('x0', 0, units='m')
    ivc.add_output('z0', 0, units='m')

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeTrajectories', ComputeTrajectories(n = n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('distance = ', p['ComputeTrajectories.x'])
    print('altitude = ', p['ComputeTrajectories.z'])
