import ortools.linear_solver.pywraplp as pywraplp

solver = pywraplp.Solver.CreateSolver('CBC')

x1 = solver.IntVar(0, solver.infinity(), 'x1')
x2 = solver.IntVar(0, solver.infinity(), 'x2')
x3 = solver.IntVar(0, solver.infinity(), 'x3')
x4 = solver.IntVar(0, solver.infinity(), 'x4')
x5 = solver.IntVar(0, solver.infinity(), 'x5')

solver.Add(x1 + 1.2*x2 - 2*x3 + 1.5*x4 + 2*x5 >= 30 )
solver.Add(- x2 + 2*x3 + x5 >= 25 )
solver.Add(2*x1 + 1.5*x2 + 3*x3 + 4*x4 >= 12.5 )

solver.Minimize(2*x1 + 5*x2 + 3*x3 + 3*x4 + 2*x5)

status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    print('x1 =', x1.solution_value())
    print('x2 =', x2.solution_value())
    print('x3 =', x3.solution_value())
    print('x4 =', x4.solution_value())
    print('x5 =', x5.solution_value())
else:
    print('The problem does not have an optimal solution.')