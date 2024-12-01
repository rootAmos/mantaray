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
        self.add_input('ub', val= np.ones(self.options['n']), desc='airspeed in x-axis of fixed body frame', units='m/s')

        # Outputs
        self.add_output('lift', val= np.ones(self.options['n']), desc='lift force', units='N')
        self.add_output('drag', val= np.ones(self.options['n']), desc='drag force', units='N')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        CL = inputs['CL']
        CD = inputs['CD']
        rho = inputs['rho']
        ub = inputs['ub']
        S = inputs['S']


        Lift = 0.5 * rho * ub**2 * S * CL
        Drag = 0.5 * rho * ub**2 * S * CD


        # Pack outputs
        outputs['lift'] = Lift
        outputs['drag'] = Drag


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('utas', 100, units='m/s')
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

    print('Lift = ', p['ComputeAeroForces.lift'])
    print('Drag = ', p['ComputeAeroForces.drag'])
