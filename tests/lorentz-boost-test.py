import numpy as np
import vector

practice = vector.array(
    {
        "E": [23],
        "px": [10],
        "py": [0],
        "pz": [0],
    }
)

boost = vector.array(
    {
        "E": [18],
        "px": [17],
        "py": [0],
        "pz": [0],
    }
)

boosted_vec = practice.boostCM_of_p4(-boost)
print(boosted_vec)