import numpy as np  
import openmdao.api as om
from functools import partial
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)



class ComputeVelocityFromCL(om.ExplicitComponent):
    """
    Compute the velocity of an aircraft from lift coefficient. Ignores influence of HTP .
    """

    def initialize(self):
        self.options.declare('g', default=9.806 , desc='gravity')
        self.options.declare('n', default=1, desc='number of data points')


    def setup(self):

        # Inputs    
        self.add_input('mass', val= 1, desc='mass', units='kg')
        self.add_input('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')        
        self.add_input('gamma', val= np.ones(self.options['n']), desc='flight path angle', units='rad')
        self.add_input('S', val= 30, desc='wing area', units='m**2')
        self.add_input('CL', val= 0.7* np.ones(self.options['n']), desc='lift coefficient', units=None)

        # Outputs
        self.add_output('vel', val= 100*np.ones(self.options['n']), desc='true airspeed', units='m/s')

    def setup_partials(self):
        self.declare_partials('vel', 'mass')
        self.declare_partials('vel', 'CL')
        self.declare_partials('vel', 'rho')
        self.declare_partials('vel', 'S')
        self.declare_partials('vel', 'gamma')


    def compute(self, inputs, outputs):

        # Unpack inputs
        mass = inputs['mass']
        rho = inputs['rho']
        gamma = inputs['gamma']
        S = inputs['S']
        CL = inputs['CL']
        
        # Unpack constants
        g = self.options['g']

        # Lift coefficient Required
        vel = (mass * g * np.cos(gamma) / ( 0.5 * rho * CL * S  ))**0.5

        # Pack outputs
        outputs['vel'] = vel

    def compute_partials(self, inputs, J):

        # Unpack inputs
        mass = inputs['mass']
        rho = np.transpose(inputs['rho'])
        S = inputs['S']
        gamma = inputs['gamma']
        CL = inputs['CL']

        # Unpack constants
        g = self.options['g']
        n = self.options['n']

        J['vel', 'mass'] =  0.5 * g * np.cos(gamma) / ( mass * ( 0.5 * rho * CL * S  ))**0.5
        J['vel', 'S'] =  -0.5 * g * np.cos(gamma) / ( mass * ( 0.5 * rho * CL * S  ))**0.5 / S
        J['vel', 'rho'] = np.eye(n) * -0.5 * g * np.cos(gamma) / ( mass * ( 0.5 * rho * CL * S  ))**0.5 / rho
        J['vel', 'gamma'] = np.eye(n) * -0.5 * g * np.sin(gamma) / ( mass * ( 0.5 * rho * CL * S  ))**0.5
        J['vel', 'CL'] = np.eye(n) * -0.5 * g * np.cos(gamma) / ( mass * ( 0.5 * rho * CL * S  ))**0.5 / CL


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    n = 10

    ivc = om.IndepVarComp()
    ivc.add_output('rho', 1.225 * np.ones(n), units='kg/m**3')
    ivc.add_output('mass', 8600, units='kg')
    ivc.add_output('gamma', 3 * np.pi/180 * np.ones(n), units='rad')
    ivc.add_output('S', 30, units='m**2')
    ivc.add_output('CL', 0.7 * np.ones(n), units=None)   

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeVelocityFromCL', ComputeVelocityFromCL(n=n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    """ 

    # setup the optimization
    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'

    p.model.add_design_var('CL', lower=0.5, upper=1)
    #p.model.add_constraint('vel', lower=30)

    # Climb to 30,000 ft
    p.model.add_objective('ComputeVelocity.vel')

    p.setup()

    # run the optimization
    p.run_driver()
    """

    print('vel = ', p['ComputeVelocityFromCL.vel'])
