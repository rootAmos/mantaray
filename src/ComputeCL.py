import numpy as np  
import openmdao.api as om



class ComputeCL(om.ExplicitComponent):
    """
    Compute the lift coefficient and angle of attack of aircraft. Ignores influence of HTP .
    """

    def iniappltialize(self):
        self.options.declare('g', default=9.806 , desc='gravity')


    def setup(self):

        # Inputs    
        self.add_input('mass', val=0, desc='mass', units='kg')
        self.add_input('vtas', val=0, desc='true airspeed', units='m/s')
        self.add_input('rho', val=0, desc='air density', units='kg/m**3')
        self.add_input('CLa', val=0, desc='lift curve slope', units='1/rad')
        self.add_input('CL0', val=0, desc='zero-angle-of-attack lift coefficient', units=None)
        self.add_input('i_inc', val=0, desc='incidence angle', units='rad')
        self.add_input('gamma', val=0, desc='flight path angle', units='rad')

        # Outputs
        self.add_output('CL', val=0, desc='lift coefficient', units=None)
        self.add_output('aofa', val=0, desc='angle of attack', units='rad')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        mass = inputs['mass']
        vtas = inputs['vtas']
        rho = inputs['rho']
        CLa = inputs['CLa']
        CL0 = inputs['CL0']
        i_inc = inputs['i_inc']
        gamma = inputs['gamma']

        # Unpack constants
        g = self.options['g']

        # Lift coefficient Required
        CL = mass * g * np.cos(gamma) / 0.5 * rho * vtas**2  

        # Angle of attack
        aofa = (CL - CL0) / CLa + i_inc

        # Pack outputs
        outputs['aofa'] = aofa
        outputs['CL'] = CL


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('mass', 8600, units='kg')
    ivc.add_output('aofa', 3 * np.pi/180, units='rad')
    ivc.add_output('vtas', 100, units='m/s')

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
