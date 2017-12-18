import numpy as np
BOUNDS = 3000
nturbs = 20
x = np.arange(BOUNDS)
y = np.arange(BOUNDS)
x = [x for _ in range(BOUNDS)]
y = [y for _ in range(BOUNDS)]

u = [[10 for _ in range(BOUNDS)] for __ in range(BOUNDS)]
udef = [[[0 for _ in range(BOUNDS)] for __ in range(BOUNDS)] for n in nturbs]

for 
