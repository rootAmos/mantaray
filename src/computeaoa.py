import numpy as np  
import openmdao.api as om



class ComputeAofA(om.ExplicitComponent):
    """
    Compute the angle of attack of aircraft. Ignores influence of HTP .
    """

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')


    def setup(self):

        # Inputs    
        self.add_input('CLa', val=1, desc='lift curve slope', units='1/rad')
        self.add_input('alpha_0', val=0, desc='zero lift angle of attack', units='rad')
        self.add_input('alpha_i', val=0, desc='incidence angle', units='rad')
        self.add_input('CL', val= 0.7* np.ones(self.options['n']), desc='lift coefficient', units=None)

        # Outputs
        self.add_output('aofa', val= np.ones(self.options['n']), desc='angle of attack', units='rad')

    def setup_partials(self):
        self.declare_partials('aofa', 'CL')
        self.declare_partials('aofa', 'CLa')
        self.declare_partials('aofa', 'alpha_0')
        self.declare_partials('aofa', 'alpha_i')

        
    def compute(self, inputs, outputs):

        # Unpack inputs
        CLa = inputs['CLa']
        alpha_0 = inputs['alpha_0']
        alpha_i = inputs['alpha_i']
        CL = inputs['CL']

        # Angle of attack
        aofa = (CL / CLa )+ alpha_0 - alpha_i

        # Pack outputs
        outputs['aofa'] = aofa

    def compute_partials(self, inputs, J):

        # Unpack inputs
        CLa = inputs['CLa']

        # Unpack constants
        J['aofa', 'CL'] = 1 / CLa
        J['aofa', 'CLa'] = -1 / CLa**2
        J['aofa', 'alpha_0'] = 1
        J['aofa', 'alpha_i'] = -1


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('CLa', 1, units='1/rad')
    ivc.add_output('alpha_0', 0, units='rad')
    ivc.add_output('alpha_i', 0, units='rad')
    ivc.add_output('CL', 0.7, units=None)

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeAofA', ComputeAofA(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('aofa = ', p['ComputeAofA.aofa'])
