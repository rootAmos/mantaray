import numpy as np  
import openmdao.api as om



class ComputeCL(om.ExplicitComponent):
    """
    Compute the lift coefficient needed for an aircraft. Ignores influence of HTP .
    """

    def initialize(self):
        self.options.declare('g', default=9.806 , desc='gravity')
        self.options.declare('n', default=1, desc='number of data points')


    def setup(self):

        # Inputs    
        self.add_input('mass', val= 1, desc='mass', units='kg')
        self.add_input('vel', val= 100*np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.add_input('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')        
        self.add_input('gamma', val= np.ones(self.options['n']), desc='flight path angle', units='rad')
        self.add_input('S', val= 30, desc='wing area', units='m**2')

        # Outputs
        self.add_output('CL', val= 0.7* np.ones(self.options['n']), desc='lift coefficient', units=None)

    def setup_partials(self):
        self.declare_partials('CL', 'mass')
        self.declare_partials('CL', 'vel')
        self.declare_partials('CL', 'rho')
        self.declare_partials('CL', 'S')
        self.declare_partials('CL', 'gamma')


    def compute(self, inputs, outputs):

        # Unpack inputs
        mass = inputs['mass']
        vel = inputs['vel']
        rho = inputs['rho']
        gamma = inputs['gamma']
        S = inputs['S']
        
        # Unpack constants
        g = self.options['g']

        # Lift coefficient Required
        cl = mass * g * np.cos(gamma) / ( 0.5 * rho * vel**2 * S  )

        # Pack outputs
        outputs['CL'] = cl

    def compute_partials(self, inputs, J):

        # Unpack inputs
        mass = inputs['mass']
        vel = inputs['vel']
        rho = inputs['rho']
        S = inputs['S']
        gamma = inputs['gamma']

        # Unpack constants
        g = self.options['g']

        J['CL', 'mass'] = g * np.cos(gamma) / ( 0.5 * rho * vel**2 * S  )
        J['CL', 'vel'] = -mass * g * np.cos(gamma) / ( 0.5 * rho * vel**3 * S  )
        J['CL', 'rho'] = -mass * g * np.cos(gamma) / ( 0.5 * rho**2 * vel**2 * S  )
        J['CL', 'S'] = -mass * g * np.cos(gamma) / ( 0.5 * rho * vel**2 * S**2  )
        J['CL', 'gamma'] = -mass * g * np.sin(gamma) / ( 0.5 * rho * vel**2 * S  )

if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('mass', 8600, units='kg')
    ivc.add_output('gamma', 3 * np.pi/180, units='rad')
    ivc.add_output('vel', 100, units='m/s')

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeCL', ComputeCL(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('CL = ', p['ComputeCL.CL'])
