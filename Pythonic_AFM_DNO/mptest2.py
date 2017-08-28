import multiprocessing as mp
import random
import string
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def plotData(outputDir, plotNum):
    outFilename = "plot_%d.pdf" % (plotNum,)
    outFilepath = os.path.join(outputDir, outFilename)
    
    # Plot some random data
    # Adapted from: http://matplotlib.org/examples/shapes_and_collections/scatter_demo.html
    N = 500
    # First we need to re-initialize the random number generator for each worker
    # See: https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    np.random.seed( int( time() ) + plotNum )
    x = np.random.rand(N)
    y = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses

    print "\tMaking plot %d" % (plotNum,) 
    plt.scatter(x, y, s=area, alpha=0.5)
    plt.savefig(outFilepath)
    # Clear figure so that the next plot this worker makes will not contain
    # data from previous plots
    plt.clf() 
    
    return (plotNum, outFilepath)
    
pool = mp.Pool( mp.cpu_count() )