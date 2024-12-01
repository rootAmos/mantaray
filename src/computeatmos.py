import numpy as np  
import openmdao.api as om



class ComputeAtmos(om.ExplicitComponent):
    """
    Compute the thrust produced by a propeller.

    References
    [1]  NASA Earth Atmosphere Model ~ https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    """

    def initialize(self):

        self.options.declare('n', default=1, desc='number of data points')

       

    def setup(self):


        # Inputs    
        self.add_input('z', val= np.ones(self.options['n']), desc='altitude', units='m')

        # Outputs
        self.add_output('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')
        self.add_output('temp', val= np.ones(self.options['n']), desc='air temperature', units='degC')
        self.add_output('pressure', val= np.ones(self.options['n']), desc='air pressure', units='Pa')
        self.add_output('c', val= np.ones(self.options['n']), desc='speed of sound', units='m/s')


        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        z = inputs['z'] # altitude  (m)


        # Temperature (degC)
        temp = np.where(z > 25000, -131.21 + 0.00299 * z, 
                        np.where(z > 11000, -56.55, 15.04 - 0.00649 * z))

        # Pressure (kPa)
        pressure = np.where(z > 25000, 2.488*( (temp+273.1) /216.6)**(-11.388),
                            np.where(z > 11000, 22.65*np.exp(1.73-0.000157*z), 
                                     101.29*((temp+273.1)/288.08)**5.256))
        
        # Density (kg/m^3)
        
        rho = pressure/(0.2869*(temp+273.1))


        # Speed of sound (m/s)
        c = (1.4*8.314/(28.96/1000)*(temp+273.1))

        outputs['rho'] = rho
        outputs['temp'] = temp
        outputs['pressure'] = pressure
        outputs['c'] = c


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('z'  , np.ones(100)*1000, units='m')




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

