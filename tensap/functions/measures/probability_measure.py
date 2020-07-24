'''
Module probability_measure.

'''

import tensap


class ProbabilityMeasure(tensap.Measure):
    '''
    Class ProbabilityMeasure.

    '''

    @staticmethod
    def mass():
        return 1

    def random_rejection(self, n, Y, c, m):
        # TODO random_rejection
        raise NotImplementedError('Method not implemented.')
