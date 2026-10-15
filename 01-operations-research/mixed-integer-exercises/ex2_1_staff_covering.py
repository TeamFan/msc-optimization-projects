import ortools.linear_solver.pywraplp as pywraplp

solver = pywraplp.Solver.CreateSolver('CBC')

x1 = solver.IntVar(0, solver.infinity(), 'x1')
x2 = solver.IntVar(0, solver.infinity(), 'x2')
x3 = solver.IntVar(0, solver.infinity(), 'x3')
x4 = solver.IntVar(0, solver.infinity(), 'x4')
x5 = solver.IntVar(0, solver.infinity(), 'x5')
x6 = solver.IntVar(0, solver.infinity(), 'x6')
x7 = solver.IntVar(0, solver.infinity(), 'x7')

solver.Add(x1 + x6 >= 15 )
solver.Add(x1 + x2 + x3 + x6 >= 45 )
solver.Add(x1 + x2 + x3 + x4 + x6 >= 45 )
solver.Add(x1 + x2 + x3 + x4 + x6 + x7 >= 55 )
solver.Add(x2 + x3 + x4 + x6 + x7 >= 30 )
solver.Add(x1 + x3 + x5 + x7 >= 30 )
solver.Add(x1 + x2 + x4 + x5 + x7 >= 45 )
solver.Add(x1 + x2 + x4 + x5 >= 45 )
solver.Add(x2 + x4 + x5 + x7 >= 45 )
solver.Add(x4 + x5 + x7 >= 25 )

solver.Minimize(7*x1 + 7*x2 + 5*x3 + 7*x4 + 5*x5 + 5*x6 + 6*x7)

status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    print('x1 =', x1.solution_value())
    print('x2 =', x2.solution_value())
    print('x3 =', x3.solution_value())
    print('x4 =', x4.solution_value())
    print('x5 =', x5.solution_value())
    print('x6 =', x6.solution_value())
    print('x7 =', x7.solution_value())
else: 
    print('The problem does not have an optimal solution.')