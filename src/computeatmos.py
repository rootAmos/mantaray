import numpy as np  
import openmdao.api as om



class ComputeAtmos(om.ImplicitComponent):
    """
    Compute the thrust produced by a propeller.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):

        self.options.declare('eta_prop', default=0.8, desc='propeller efficiency')
        self.options.declare('n', default=1, desc='number of data points')

       

    def setup(self):

        # Inputs    
        self.add_input('z', val= np.ones(self.options['n']), desc='altitude', units='m')
        self.add_input('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')

        # Outputs
        self.add_output('total_thrust_gen', val=0, desc='total aircraft thrust generated', units='N')


        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        z = inputs['z'] # altitude  (m)

        if z > 25000:
            pressure = 2.488*( (z+273.1) /216.6)^(-11.388)
            temp = -131.21 + 0.00299 * z
        elif z > 11000:
            pressure = 22.65*np.exp(1.73-0.000157*z)
            temp = -56.55 
        else:
            pressure = 101.29*((z+273.1)/288.08)^5.256
            temp = 15.04 - 0.00649 * z

        rho = pressure/(0.2869*(temp+273.1))

        outputs['rho'] = rho


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('z', 1000, units='m')




    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeAtmos', ComputeAtmos(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('rho = ', p['ComputeAtmos.rho'])

