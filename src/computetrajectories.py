import numpy as np  
import openmdao.api as om



class ComputeTrajectories(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')        


    def setup(self):

        # Inputs 
        self.add_input('z0', val=  0, desc='initial distance in z-dir of earth fixed body frame', units='m')
        self.add_input('x0', val=  0, desc='initial distance in x-dir of earth fixed body frame', units='m')
        self.add_input('vx', val= np.ones(self.options['n']), desc='horizontal velocity in the earth frame', units='m/s')
        self.add_input('vz', val= np.ones(self.options['n']), desc='vertical velocity in the earth frame', units='m/s')
        self.add_input('dt', val= np.ones(self.options['n']), desc='time step', units='s')


        # Outputs
        self.add_output('x', val= np.ones(self.options['n']), desc='trajectory in the x-axis', units='m')
        self.add_output('z', val= np.ones(self.options['n']), desc='trajectory in the z-axis', units='m')


    def setup_partials(self):

        self.declare_partials('x', 'vx')
        self.declare_partials('x', 'dt')
        self.declare_partials('x', 'x0')
        self.declare_partials('z', 'vz')
        self.declare_partials('z', 'dt')
        self.declare_partials('z', 'z0')


    def compute(self, inputs, outputs):

        # Unpack inputs
        x0 = inputs['x0']
        z0 = inputs['z0']
        vx = inputs['vx']
        vz = inputs['vz']
        dt = inputs['dt']

        # Unpack options
        n = self.options['n']

        # Compute distance, time, and altitude
        x = np.cumsum(vx * dt) + x0
        z = np.cumsum(vz * dt) + z0

        # Pack outputs
        outputs['x'] = x
        outputs['z'] = z

    def compute_partials(self, inputs, J):

        # Unpack inputs
        vx = inputs['vx']
        vz = inputs['vz']
        dt = inputs['dt']

        J['x', 'vx'] = dt
        J['x', 'dt'] = vx
        J['x', 'x0'] = 1


        J['z', 'vz'] = dt
        J['z', 'dt'] = vz
        J['z', 'z0'] = 1



    

if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    n = 100

    ivc = om.IndepVarComp()
    ivc.add_output('vx', 100 * np.ones(n), units='m/s')
    ivc.add_output('vz', 3 * np.ones(n), units='m/s')
    ivc.add_output('dt', 0.1 * np.ones(n), units='s')

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
