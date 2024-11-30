import numpy as np  
import openmdao.api as om



class ComputeAeroForces(om.ExplicitComponent):

    def iniappltialize(self):
        pass


    def setup(self):

        # Inputs    
        self.add_input('CL', val=0, desc='lift coefficient', units=None)
        self.add_input('CD', val=0, desc='drag coefficient', units=None)
        self.add_input('vtas', val=0, desc='true airspeed', units='m/s')
        self.add_input('rho', val=0, desc='air density', units='kg/m**3')

        # Outputs
        self.add_output('Lift', val=0, desc='lift force', units='N')
        self.add_output('Drag', val=0, desc='drag force', units='N')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        CL = inputs['CL']
        CD = inputs['CD']
        rho = inputs['rho']
        vtas = inputs['vtas']


        Lift = 0.5 * rho * vtas**2 * CL
        Drag = 0.5 * rho * vtas**2 * CD


        # Pack outputs
        outputs['Lift'] = Lift
        outputs['Drag'] = Drag


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('vtas', 100, units='m/s')
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

    print('Lift = ', p['ComputeAeroForces.Lift'])
    print('Drag = ', p['ComputeAeroForces.Drag'])
