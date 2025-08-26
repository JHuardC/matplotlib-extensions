"""
Provide example use case of raincloud plot.
"""
### Imports
from pathlib import Path
import dotenv
from numpy.random import default_rng
from numpy import append, reshape
import matplotlib.pyplot as plt
from pandas import DataFrame
from mpl_ext import pyplot_raincloud

PROJECT_PATH = Path(dotenv.find_dotenv()).absolute().parent

if __name__ == '__main__':

    rng = default_rng()

    x = append(
        rng.normal(loc = 7, scale = .5, size = 200),
        rng.gamma(shape = 5, scale = 1, size = 200)
    )

    x = DataFrame(reshape(x, (-1, 1)), columns = ['x'])

    fig, ax = plt.subplots()

    cloud = pyplot_raincloud(x, 'x', ax)

    ax.set_title('Example Raincloud')
    
    fig.savefig(
        PROJECT_PATH.joinpath('tests', 'test_outputs', 'raincloud_example.png')
    )