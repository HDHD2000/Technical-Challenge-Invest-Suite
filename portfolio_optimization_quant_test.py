import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##QUESTION 1:

##The goal is to find the optimal portfolio weights for the S&P500 in the given time frame and which minimize the expected protfolio risk
##We use the expected portfolio risk as measure of risk in this case
##The goal is to find the weights minimising the portfolio risk for the given data
##This comes down to a quadratic minimisation problem: this is the mean-varianceoptimisation approach
##We use the code and library from https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/2-Mean-Variance-Optimisation.ipynb for this implementation

import pypfopt

##let us first import the data

data = pd.read_excel(io='\Users\hugo\Documents\Non Backed-up Files\QUANT Test\S&P500.xlsx')

##let us take the data we want: the end of each month between December 2014 and Dewcember 2019



##in order to find the weights we need to identify the covariance matrix associated to the given data

from pypfopt import risk_models
from pypfopt import plotting

data_cov = risk_models.sample_cov(data)

##we then use the pyportopt library to find the optimal portfolio weights for the given portfolio minimisation problem

from pypfopt import EfficientFrontier

S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
ef = EfficientFrontier(None, S, weight_bounds=(0, None))
ef.min_volatility()
weights = ef.clean_weights()

##This yields the desired weights minimising the portfolio risk for the given data

##==================================================##

##QUESTION 2:

##One hopes that the portfolio created in question 1 has a lower risk than the equally weighted portfolio strategy.

##

##===================================================##

##QUESTION 3:

##I have never studied or used any of these terms before and therefore cannot provide any insight into their advantages or disadvantages.