import numpy as np  
import openmdao.api as om



class ComputeKinematics(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')        


    def setup(self):

        # Inputs    
        self.add_input('gamma', val= np.ones(self.options['n']) * 0, desc='flight path angle', units='rad')
        self.add_input('vtas', val= np.ones(self.options['n']) * 0, desc='true airspeed', units='m/s')
        self.add_input('dt', val= np.ones(self.options['n']) * 0, desc='time step', units='s')
        self.add_input('x0', val=  0, desc='initial distance', units='m')
        self.add_input('z0', val=  0, desc='initial altitude', units='m')
        self.add_input('t0', val=  0, desc='initial time', units='s')


        # Outputs
        self.add_output('t', val=np.ones(self.options['n']), desc='time', units='s')
        self.add_output('t1', val=0, desc='end time', units='s')
        self.add_output('x', val= np.ones(self.options['n']), desc='trajectory in the x-axis', units='m')
        self.add_output('z', val= np.ones(self.options['n']), desc='trajectory in the z-axis', units='m')
        self.add_output('x1', val=0, desc='end position in the x-axis', units='m')
        self.add_output('z1', val=0, desc='end position in the z-axis', units='m')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        gamma = inputs['gamma']
        vtas = inputs['vtas']
        dt = inputs['dt']
        x0 = inputs['x0']
        z0 = inputs['z0']

        # Compute horizontal and vertical velocities
        vx = vtas * np.cos(gamma)
        vz = vtas * np.sin(gamma)

        # Compute distance, time, and altitude
        x = np.cumsum(vx * dt) + x0
        t = np.cumsum(dt)
        z = np.cumsum(vz * dt) + z0

        # Pack outputs
        outputs['t'] = t
        outputs['x'] = x
        outputs['z'] = z
        outputs['t1'] = t[-1]
        outputs['x1'] = x[-1]
        outputs['z1'] = z[-1]

if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('vtas', 100, units='m/s')
    ivc.add_output('gamma', 0, units='rad')
    ivc.add_output('dt', 0.1, units='s')
    ivc.add_output('x0', 0, units='m')
    ivc.add_output('z0', 0, units='m')

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeKinematics', ComputeKinematics(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('distance = ', p['ComputeKinematics.distance'])
    print('altitude = ', p['ComputeKinematics.altitude'])
    print('time = ', p['ComputeKinematics.time'])
