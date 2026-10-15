import ortools.linear_solver.pywraplp as pywraplp

solver = pywraplp.Solver.CreateSolver('CBC')

x1 = solver.IntVar(0, 1, 'x1')
x2 = solver.IntVar(0, 1, 'x2')
x3 = solver.IntVar(0, 1, 'x3')
x4 = solver.IntVar(0, 1, 'x4')

solver.Add(6*x1 + 3*x2 + 5*x3 + 2*x4 <= 10 )
solver.Add(x3 + x4 <= 1 )
solver.Add(-x1 + x3 <= 0 )
solver.Add(-x2 + x4 <= 0 )

solver.Maximize(9*x1 + 5*x2 + 6*x3 + 4*x4 )

status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    print('x1 =', x1.solution_value())
    print('x2 =', x2.solution_value())
    print('x3 =', x3.solution_value())
    print('x4 =', x4.solution_value())
else:
    print('The problem does not have an optimal solution.')