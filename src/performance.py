import numpy as np
import matplotlib
import pandas as pd
import scipy
import openmdao.api as om
from openmdao.api import NewtonSolver, DirectSolver, ArmijoGoldsteinLS, Problem, Group


from src.computecl import ComputeCL
from src.computekinematics import ComputeKinematics
from src.computeaeroforces import ComputeAeroForces
from src.computeturbine import ComputeTurbine
from src.computebatterydischarge import ComputeBatteryDischarge
from src.computeaero import ComputeAero
from src.computepropthrustgen import ComputePropThrustGen   
from src.computevelocities import ComputeVelocities
from src.computeatmos import ComputeAtmos

"""
A python tool to size electric power trains and electric subsystems for aircraft.

Copyright Bauhaus Luftfahrt e.V.
"""

# Futures

# Built-in/Generic Imports

class Performance(om.Group):
    """
    Group containing the Stack MDA without IndepVarComp for inputs
    """

    def initialize(self):
        self.options.declare('n', default=1, desc='number of points')
        self.options.declare('g', default=9.806, desc='gravitational acceleration')
        self.options.declare('wave_a', default=0.825, desc='wave drag a term')
        self.options.declare('wave_b', default=2.61, desc='wave drag b term')
        self.options.declare('MCrit', default=0.9, desc='critical mach number')

    def setup(self):

        self.add_subsystem(name='ComputeAtmos',
                           subsys=ComputeAtmos(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ComputeAero',
                           subsys=ComputeAero(
                               n=self.options['n'], 
                               wave_a=self.options['wave_a'], 
                               wave_b=self.options['wave_b'], 
                               MCrit=self.options['MCrit']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        self.add_subsystem(name='ComputeKinematics',
                           subsys=ComputeKinematics(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ComputeVelocities',
                           subsys=ComputeVelocities(n=self.options['n'], g=self.options['g']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ComputeCL',
                           subsys=ComputeCL(n=self.options['n'], g=self.options['g']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem(name='ComputePropThrustGen',
                           subsys=ComputePropThrustGen(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])  
        
        self.add_subsystem(name='ComputeAeroForces',
                           subsys=ComputeAeroForces(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])  
        
        self.add_subsystem(name='ComputeTurbine',
                           subsys=ComputeTurbine(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ComputeBatteryDischarge',
                           subsys=ComputeBatteryDischarge(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])        


        self.nonlinear_solver = NewtonSolver()
        self.linear_solver = DirectSolver()
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 200
        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.options['stall_limit'] = 4
        self.nonlinear_solver.linesearch = om.BoundsEnforceLS()


if __name__ == "__main__":
    import numpy as np
    # from omxdsm import write_xdsm

    # generate path where xlsx with all data can be found
    n = 10

    ivc = om.IndepVarComp()

    mass =  8600 # mass flow in to compressor
    S =  30 # wing area
    b = 20 # wing span
    AR = b**2 / S # wing aspect ratio



    # Prepare initial values

    # Position
    ivc.add_output('x0', val=0, units='m', desc='initial position')
    ivc.add_output('z0', val=0, units='m', desc='initial altitude')
    ivc.add_output('t0', val=0, units='s', desc='initial time')

    # Velocity
    ivc.add_output('u0', val=50, units='m/s', desc='initial velocity in body fixed axis x direction')
    ivc.add_output('w0', val=0, units='m/s', desc='initial velocity in body fixed axis z direction')
    ivc.add_output('gamma', val=0 * np.ones(n), units='rad', desc='flight path angle')

    # Time step
    ivc.add_output('dt', val=0.1 * np.ones(n), units='s', desc='time step')

    # Aircraft geometry
    ivc.add_output('S', val=S , units='m**2', desc='wing area')
    ivc.add_output('AR', val=AR , units=None, desc='wing aspect ratio')
    ivc.add_output('mass', val=mass, units='kg', desc='aircraft mass')


    # Aero
    ivc.add_output('alpha_0', val=-3 * np.pi / 180 , units='rad', desc='zero-lift angle of attack')   
    ivc.add_output('alpha_i', val=2 * np.pi / 180 , units='rad', desc='incidence angle')   
    ivc.add_output('CLa', val=5.5, units='1/rad', desc='lift curve slope')
    ivc.add_output('Cd0', val=0.02, units=None, desc='zero-lift drag coefficient')
    ivc.add_output('e', val=0.8, units=None, desc='oswald efficiency factor')

    # Powertrain
    ivc.add_output('eta_batt', val=0.9, units=None, desc='battery efficiency')
    ivc.add_output('eta_motor', val=0.9, units=None, desc='motor efficiency')
    ivc.add_output('eta_pe', val=0.9, units=None, desc='power electronics efficiency')
    ivc.add_output('eta_cbl', val=0.9, units=None, desc='cables efficiency')
    ivc.add_output('eta_prop', val=0.8, units=None, desc='propeller efficiency')
    ivc.add_output('eta_duct', val=0.8, units=None, desc='duct efficiency')
    ivc.add_output('eta_fan', val=0.8, units=None, desc='fan efficiency')   

    ivc.add_output('batt_cap', val=1000 * 3600, units='J', desc='battery capacity')
    ivc.add_output('soc_0', val=1, units=None, desc='initial battery state of charge')
    ivc.add_output('num_engines', val=2, units=None, desc='number of engines')
    ivc.add_output('d_blade', val=1, units='m', desc='blade diameter')
    ivc.add_output('d_hub', val=0.2, units='m', desc='hub diameter')

    ivc.add_output('unit_shaft_pow_gen', val=500e3 * np.ones(n), units='W', desc='power generated per engine')
    


    
    p = om.Problem()
    model = p.model
    p.model.add_subsystem('ivc', ivc, promotes=['*'])
    p.model.add_subsystem('System', Performance(n=n),
                          promotes_inputs=['*'])

    # set model settings
    p.model.nonlinear_solver = om.NewtonSolver()
    # self.linear_solver = om.LinearRunOnce()
    p.model.linear_solver = om.DirectSolver()

    p.model.nonlinear_solver.options['iprint'] = 2
    p.model.nonlinear_solver.options['maxiter'] = 200
    p.model.nonlinear_solver.options['solve_subsystems'] = True
    p.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()

    p.setup()
    om.n2(p)


    # Create a recorder variable
    # recorder = om.SqliteRecorder('cases.sql')
    # Attach a recorder to the problem
    # p.add_recorder(recorder)

    p.run_model()

    # Instantiate your CaseReader
    # cr = om.CaseReader("cases.sql")
    # Isolate "problem" as your source
    # driver_cases = cr.list_cases('problem', out_stream=None)
    # Get the first case from the recorder
    # case = cr.get_case('after_run_driver')

    # These options will give outputs as the model sees them
    # Gets value but will not convert units

    # print(case.list_outputs())

    # #om.view_connections(p, outfile="fuel_cell_system.html", show_browser=True)
    # write_xdsm(p, filename='bop',show_browser=True, out_format='pdf',
    #        quiet=False, output_side='left', recurse=False)
