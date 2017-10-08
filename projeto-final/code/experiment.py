class Experiment(object):
    
    def __init__(self, rbm):
        self.rbm = rbm

    def len_of_pedalboard(self, pedalboard):
        return len(pedalboard[pedalboard == True])

    def recommend(self, pedalboard, score=-.08, size=None, maximum_iterations=10000, minimum_iterations=10):
        """
        Recommend a pedalboard with a score of a sample less then the defined score
        
        :param pedalboard: Part of a pedalboard, contains the initial audio plugins
        :param score: Minimum score of the recommendation
        :param size: Pedalboard recommended needs the same size defined
                     or not, if size is None
        :param maximum_iterations: Maximum of gibbs sampling for discover a recommendation
                                   if a pedalboard aren't discovered with < score defined,
                                   then is returned the pedalboard with the bigger score
        :param minimum_iterations: Is necessary executes at least minimum_iterations
        """
        correct_size = lambda pedalboard: True if size is None else self.len_of_pedalboard(pedalboard) == size
        current_iteration = 0
        
        recommendation = pedalboard
        best_recommendation = {'pedalboard': pedalboard, 'score': -10000}
        
        while self._recommend_condition(recommendation, current_iteration, minimum_iterations, maximum_iterations, score, size):
            recommendation |= pedalboard
            recommendation = self.rbm.gibbs(recommendation)

            score_recommendation = self.score_sample(recommendation)
            if correct_size(recommendation) and score_recommendation > best_recommendation['score']:
                best_recommendation['pedalboard'] = recommendation.copy()
                best_recommendation['score'] = score_recommendation

            current_iteration += 1

        return best_recommendation['pedalboard'] if current_iteration == maximum_iterations else recommendation
    
    def _recommend_condition(self, pedalboard, current_iteration, minimum_iterations, maximum_iterations, score, size):
        if current_iteration < minimum_iterations:
            return True
        
        correct_size = lambda: True if size is None else self.len_of_pedalboard(pedalboard) == size
        
        if self.score_sample(pedalboard) > score and correct_size():
            return False
        
        if current_iteration == maximum_iterations:
            return False
        
        return True

    def score_sample(self, sample):
        """
        Compute the pseudo-likelihood of the sample
        :param sample: Pedalboard
        :return: score sample
        """
        return self.rbm.score_samples([sample])

    def evaluate(self, recommended, train, test, original):
        """
        Returns the tax of:
         - successful preserved audio plugins
         - successful discovered plugins
         - correct audio plugins (based in original pedalboard)

        :param recommended: Pedalboard recommended
        :param train: Pedalboard used for recommends
        :param test: original union ¬train
        :param original: Pedalboard original
        :return:
        """
        preserve = 0 if self.len_of_pedalboard(train) == 0 else self.len_of_pedalboard(train & recommended) / self.len_of_pedalboard(train)
        discover = 0 if self.len_of_pedalboard(test) == 0 else self.len_of_pedalboard(test & recommended) / self.len_of_pedalboard(test)
        total = 0 if self.len_of_pedalboard(original) == 0 else self.len_of_pedalboard(original & recommended) / self.len_of_pedalboard(original)

        return preserve, discover, total
