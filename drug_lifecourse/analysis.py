__author__ = 'thor'

from numpy import *
import numpy as np
import pandas as pd

from misc.drug_lifecourse.data_flow import DrugLifeCourse
from ut.ppi.pot import Pot
from ut.ppi.pot import ProbPot

dlc = DrugLifeCourse()


class JointAnalysis(object):
    def __init__(self, data_name='drug_use_and_social_binary_data'):
        self.dlc = DrugLifeCourse()
        self.d = self.dlc.get_data(data_name)
        self.index_vars = ['id', 'year', 'age']
        self.avars = list(set(self.d).difference(self.index_vars))
        self.count_pot = None
        self.prob_pot = None
        self.relative_risk_df = None

    def get_count_pot(self, avars):
        avars = avars or self.avars
        n_vars = len(avars)
        count_pot = dict()
        for i in range(n_vars-1):
            for j in range(i+1, n_vars):
                count_pot = dict(count_pot,
                                 **{(avars[i], avars[j]): Pot.from_points_to_count(self.d[[avars[i], avars[j]]])})
                count_pot = dict(count_pot,
                                 **{(avars[j], avars[i]): Pot.from_points_to_count(self.d[[avars[j], avars[i]]])})
        self.count_pot = count_pot
        return count_pot

    def get_prob_pot(self, avars=None):
        if self.count_pot is None:
            self.get_count_pot(avars=avars)
        self.prob_pot = {k: ProbPot(v/[]) for (k, v) in self.count_pot.iteritems()}
        return self.prob_pot

    def get_relative_risk_df(self, avars=None):
        if self.prob_pot is None:
            self.get_prob_pot(avars=avars)
        n_vars = len(self.avars)
        relrisk = list()
        for k, v in self.prob_pot.iteritems():
            relrisk_val = ProbPot(v.tb).relative_risk(k[0], k[1]).tb.pval
            if len(relrisk_val) >= 1:
                relrisk_val = relrisk_val[0]
            else:
                relrisk_val = nan
            counts = self.count_pot[k]
            exposure_counts = self.count_pot[k] >> k[1]
            relrisk.append({'event': k[0], 'exposure': k[1], 'relative_risk': relrisk_val,
                            'non_exposure_count': exposure_counts.pval_of({k[1]: 0}),
                            'exposure_count': exposure_counts.pval_of({k[1]: 1}),
                            'x, v': counts.pval_of({k[1]: 1, k[0]: 1}),
                            '~x, v': counts.pval_of({k[1]: 0, k[0]: 1}),
                            'x, ~v': counts.pval_of({k[1]: 1, k[0]: 0}),
                            '~x, ~v': counts.pval_of({k[1]: 0, k[0]: 0})
            })
        relrisk = pd.DataFrame(relrisk)
        relrisk['log_relative_risk'] = np.log2(relrisk['relative_risk'])
        def relative_risk_percentage(x):
            xx = x
            if xx >= 1:
                return xx - 1
            else:
                return 1 - (1 / xx)
        relrisk['relative_risk_percentage'] = map(relative_risk_percentage, relrisk['relative_risk'])
        self.relative_risk_df = relrisk[['exposure', 'event', 'relative_risk', 'non_exposure_count', 'exposure_count',
                                         'log_relative_risk', 'relative_risk_percentage',
                                         'x, v', '~x, v', 'x, ~v', '~x, ~v']]
        return self.relative_risk_df

