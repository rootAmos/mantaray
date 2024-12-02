import numpy as np  
import openmdao.api as om



class ComputeAeroForces(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')  


    def setup(self):

        # Inputs    
        self.add_input('CL', val= np.ones(self.options['n']), desc='lift coefficient', units=None)
        self.add_input('CD', val= np.ones(self.options['n']), desc='drag coefficient', units=None)
        self.add_input('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')
        self.add_input('S', val=1, desc='wing area', units='m**2')
        self.add_input('vel', val= np.ones(self.options['n']), desc='airspeed in x-axis of fixed body frame', units='m/s')

        # Outputs
        self.add_output('lift', val= np.ones(self.options['n']), desc='lift force', units='N')
        self.add_output('drag', val= np.ones(self.options['n']), desc='drag force', units='N')


    def setup_partials(self):
        self.declare_partials('lift', 'CL')
        self.declare_partials('lift', 'vel')
        self.declare_partials('lift', 'S')
        self.declare_partials('lift', 'rho')    
        self.declare_partials('drag', 'CD')
        self.declare_partials('drag', 'vel')
        self.declare_partials('drag', 'S')
        self.declare_partials('drag', 'rho')

    def compute(self, inputs, outputs):

        # Unpack inputs
        CL = inputs['CL']
        CD = inputs['CD']
        rho = inputs['rho']
        vel = inputs['vel']
        S = inputs['S']


        Lift = 0.5 * rho * vel**2 * S * CL
        Drag = 0.5 * rho * vel**2 * S * CD


        # Pack outputs
        outputs['lift'] = Lift
        outputs['drag'] = Drag

    def compute_partials(self, inputs, J):

        # Unpack inputs
        rho = inputs['rho']        
        CL = inputs['CL']
        CD = inputs['CD']
        vel = inputs['vel']
        S = inputs['S']

        J['lift', 'CL'] = 0.5 * rho * vel**2 * S
        J['lift', 'vel'] = 0.5 * rho * 2 * S * CL * vel
        J['lift', 'S'] = 0.5 * rho * vel**2 * CL
        J['lift', 'rho'] = 0.5 * vel**2 * S * CL

        J['drag', 'CD'] = 0.5 * rho * vel**2 * S
        J['drag', 'vel'] = 0.5 * rho * 2 * S * CD
        J['drag', 'S'] = 0.5 * rho * vel**2 * CD
        J['drag', 'rho'] = 0.5 * vel**2 * S * CD


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('vel', 100, units='m/s')
    ivc.add_output('CL', 0.5, units=None)   
    ivc.add_output('CD', 0.01, units=None)

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeAeroForces', ComputeAeroForces(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()
    #om.n2(p)
    p.run_model()

    # Setup Optimization
    # p.driver = om.ScipyOptimizeDriver()
    # p.driver.options['optimizer'] = 'SLSQP'

    # p.model.add_design_var('vel', lower=0, upper=300)
    #p.model.add_constraint('drag', lower=0)
    # p.model.add_objective('ComputeAeroForces.drag')

    # p.setup()
    # p.run_driver()

    print('Lift = ', p['ComputeAeroForces.lift'])
    print('Drag = ', p['ComputeAeroForces.drag'])
