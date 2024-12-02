import numpy as np  
import openmdao.api as om



class ComputeKinematics(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')        


    def setup(self):

        # Inputs    
        self.add_input('gamma', val= np.ones(self.options['n']) * 0, desc='flight path angle', units='rad')
        self.add_input('vel', val= np.ones(self.options['n']) * 0, desc='true airspeed', units='m/s')


        # Outputs
        self.add_output('vx', val= np.ones(self.options['n']), desc='horizontal velocity in the earth frame', units='m/s')
        self.add_output('vz', val= np.ones(self.options['n']), desc='vertical velocity in the earth frame', units='m/s')

    def setup_partials(self):
        self.declare_partials('vx', 'vel')
        self.declare_partials('vx', 'gamma')
        self.declare_partials('vz', 'vel')
        self.declare_partials('vz', 'gamma')

    def compute(self, inputs, outputs):

        # Unpack inputs
        gamma = inputs['gamma']
        vel = inputs['vel']

        # Compute horizontal and vertical velocities in earth frame
        vx = vel * np.cos(gamma)
        vz = vel * np.sin(gamma)
        
        # Pack outputs
        outputs['vx'] = vx
        outputs['vz'] = vz

    def compute_partials(self, inputs, J):

        # Unpack inputs
        gamma = inputs['gamma']
        vel = inputs['vel']

        # Compute partials
        J['vx', 'vel'] = np.cos(gamma)
        J['vx', 'gamma'] = -vel * np.sin(gamma)
        J['vz', 'vel'] = np.sin(gamma)
        J['vz', 'gamma'] = vel * np.cos(gamma)


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('vel', 100, units='m/s')
    ivc.add_output('gamma', 0, units='rad')

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

    print('horizontal velocity = ', p['ComputeKinematics.vx'])
    print('vertical velocity = ', p['ComputeKinematics.vz'])

