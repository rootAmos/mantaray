import numpy as np  
import openmdao.api as om



class ComputePropEta(om.ExplicitComponent):
    """
    Compute the power required for a propeller.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):

        self.options.declare('n', default=1, desc='number of data points')

       

    def setup(self):

        # Inputs    
        self.add_input('vel', val= np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.add_input('v_ind', val= np.ones(self.options['n']), desc='induced velocity', units='m/s')

        # Outputs
        self.add_output('eta_prplsv', val= np.ones(self.options['n']), desc='propulsive efficiency', units=None)


    def setup_partials(self):
        self.declare_partials('eta_prplsv', 'vel')
        self.declare_partials('eta_prplsv', 'v_ind')


    def compute(self, inputs, outputs):

        # Unpack inputs    
        vel = inputs['vel']
        v_ind = inputs['v_ind']

        # Compute station 3 velocity [1] Eq 15-73
        v3 = vel + 2 * v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv = 2 / (1 + v3/vel)    

        outputs['eta_prplsv'] = eta_prplsv

    def compute_partials(self, inputs, J):  

        # Unpack inputs
        vel = inputs['vel']
        v_ind = inputs['v_ind']

        # Unpack constants
        n = self.options['n']

        # Compute station 3 velocity [1] Eq 15-73
        v3 = vel + 2 * v_ind  


        J['eta_prplsv', 'vel'] = np.eye(n) * 4 * v_ind/(2 * vel + 2 * v_ind)**2
        J['eta_prplsv', 'v_ind'] = np.eye(n) -4 * vel/(2 * vel + 2 * v_ind)**2
        


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    n = 10

    ivc = om.IndepVarComp()
    ivc.add_output('vel', 100 * np.ones(n), units='m/s')
    ivc.add_output('v_ind', 10 * np.ones(n), units='m/s')


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputePropEta', ComputePropEta(n = n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('eta_prplsv = ', p['ComputePropEta.eta_prplsv'])

