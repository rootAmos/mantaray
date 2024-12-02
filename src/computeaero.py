import numpy as np  
import openmdao.api as om



class ComputeAero(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')

    def setup_partials(self):
        self.declare_partials('CD', 'CL')
        self.declare_partials('CD', 'Cd0')
        self.declare_partials('CD', 'e')
        self.declare_partials('CD', 'AR')

    def setup(self):

        # Inputs    
        self.add_input('Cd0', val=1, desc='zero-lift drag coefficient', units=None)
        self.add_input('e', val=1, desc='oswald efficiency factor', units=None)
        self.add_input('AR', val=1, desc='aspect ratio', units=None)
        self.add_input('CL', val= np.ones(self.options['n']), desc='lift coefficient', units=None)

        # Outputs
        self.add_output('CD', val= np.ones(self.options['n']), desc='drag coefficient', units=None)


    def compute(self, inputs, outputs):

        # Unpack inputs
        CL = inputs['CL']
        Cd0 = inputs['Cd0']
        e = inputs['e']
        AR = inputs['AR']

        CD = Cd0 + CL**2 / (np.pi * e * AR) 

        # Pack outputs
        outputs['CD'] = CD

    def compute_partials(self, inputs, J):

        CL = inputs['CL']
        Cd0 = inputs['Cd0']
        e = inputs['e']
        AR = inputs['AR']

        J['CD', 'CL'] = 2 * CL  / (np.pi * e * AR)
        J['CD', 'Cd0'] = 1
        J['CD', 'e'] = -CL**2 / (np.pi * e**2 * AR)
        J['CD', 'AR'] = -CL**2 / (np.pi * e * AR**2)

if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('Cd0', 0.01, units=None)
    ivc.add_output('e', 0.8, units=None)
    ivc.add_output('AR', 10, units=None)
    ivc.add_output('CL', 0.5, units=None)   

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeAero', ComputeAero(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('CD = ', p['ComputeAero.CD'])
