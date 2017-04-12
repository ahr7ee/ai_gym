# load the pickeled pixels array in the file calling this function
import numpy as np
NROWS = observation.shape[0]
NCOLS = observation.shape[1]
def get_location(observation, pixels):
    for i in xrange(NROWS):
        for j in xrange(NCOLS):
            if (i, j) not in pixels and np.array_equal(observation[i][j], [104L,72L,198L]):
            # blue pixel encountered that is not a barrier
                return (i, j) # top pixel of pinball
    return (NROWS - 1, NCOLS - 1) # something went wrong            