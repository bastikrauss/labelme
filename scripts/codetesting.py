import timeit
import torch

def test_dextr():
    SETUP_CODE = '''
from dextrs.predict import DextrModel
import numpy as np
dextrmodel = None
filename = "C:/Users/krseba/Documents/apc2016_obj3.jpg"
extreme_points_ori = np.array([[564,237],[507,287],[570,353],[617,296]])
'''
    TEST_CODE = '''
if dextrmodel is None:
    dextrmodel = DextrModel()
contour = dextrmodel.dextrPrediction(filename, extreme_points_ori)
    '''
    # timeit statement 
    time = timeit.timeit(setup = SETUP_CODE, stmt = TEST_CODE, number = 20)
    # priniting exec. time
    print('Dextr time: {}'.format(time))

if __name__ == "__main__":
    test_dextr()
    # if torch.cuda.is_available():
    #     print("Cuda available")
    # else:
    #     print("Cuda not available")
