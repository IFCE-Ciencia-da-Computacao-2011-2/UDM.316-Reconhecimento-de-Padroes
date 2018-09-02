import pandas as pd
from sklearn.model_selection import train_test_split
from code.experiment import Experiment


def generate_rbm(n_iter=5000, RANDOM_STATE=None, verbose=False):
    from code.bernoulli_rbm import BernoulliRBM

    rbm = BernoulliRBM(random_state=RANDOM_STATE, verbose=verbose)

    rbm.batch_size = 250
    rbm.learning_rate = 0.08
    rbm.n_iter = n_iter
    rbm.n_components = 50 #400 # hidden units

    return rbm


def read_data():
    class NONE:
        index = 107
        name = 'None'

    plugins = pd.read_csv('data/plugin-category.csv', index_col='id').sort_index()
    pedalboards = pd.read_csv('data/pedalboard-plugin-bag-of-words.csv', index_col=['index', 'id'])

    # Alterar o nome da coluna de plugin
    plugins_copy = plugins.copy()
    plugins_copy['Plugin'] = plugins_copy.name
    plugins_copy['Category'] = plugins_copy.category

    pedalboards.columns = [plugins_copy.Category, plugins_copy.Plugin]

    # Remover None
    pedalboards[NONE.name, NONE.name] = 0
    return pedalboards > 0, plugins


def split_pedalboard(pedalboard, plugins, train_size=.5, random_state=None):
    """
    Split pedalboard into pedalboard train and pedalboard test
    pedalboard_train contains the audio plugins that a recommendation system will receive
    pedalboard_test contains the audio_plugins that a recommendation system needs discover

    :param pedalboard: Pedalboard
    :param plugins: Audio plugins
    :param train_size: Size (in float) of the pedalboard_train
    :param random_state:

    :return: pedalboard_train, pedalboard_test
    """
    pedalboard = pedalboard.values.copy()

    index = plugins[pedalboard == True].index
    train, test = train_test_split(index, train_size=train_size, random_state=random_state)

    pedalboard_train = pedalboard.copy()
    pedalboard_test = pedalboard.copy()
    pedalboard_train[test] = 0
    pedalboard_test[train] = 0

    return pedalboard_train, pedalboard_test


def plugnis_categories_of(pedalboard, plugins):
    return plugins[pedalboard == True]


def mensurate(rbm, pedalboards, plugins):
    experiment = Experiment(rbm)
    results_train = pd.DataFrame(columns=('preserve', 'discover', 'total'))

    for position, pedalboard_index in enumerate(pedalboards.index):
        pedalboard = pedalboards.iloc[position].copy()
        train, test = split_pedalboard(pedalboard, plugins, train_size=3/3)

        recommended = experiment.recommend(train, score=-1)
        results_train.loc[pedalboard_index[0]] = experiment.evaluate(recommended, train, test, pedalboard)

    return results_train


#pedalboard = pedalboards_train.iloc[214]
#train, test = split_pedalboard(pedalboard, plugins)

#print('Pedalboard')
#print(lugnis_categories_of(plugins, pedalboard.values))
#print('Train')
#print(category_of_pedalboard(plugins, train))

#experiment = Experiment(rbm)
#recommended = experiment.recommend(train, size=experiment.len_of_pedalboard(pedalboard))

#print(lugnis_categories_of(plugins, recommended))
#print(experiment.evaluate(recommended, train, test, pedalboard))

def process(total, verbose=False, RANDOM_STATE=None):
    '''
    rbm = generate_rbm(total, RANDOM_STATE, verbose)
    pedalboards, plugins = read_data()

    pedalboards_train, pedalboards_test = train_test_split(pedalboards, train_size=.8, random_state=RANDOM_STATE)
    rbm.fit(pedalboards_train)
    print('learned')

    result_train = mensurate(rbm, pedalboards_train, plugins)
    print(result_train.mean())
    return
    '''

    for i in range(30):
        print('Iteration', i)
        rbm = generate_rbm(total, RANDOM_STATE, verbose)
        pedalboards, plugins = read_data()

        pedalboards_train, pedalboards_test = train_test_split(pedalboards, train_size=.8, random_state=RANDOM_STATE)
        rbm.fit(pedalboards_train)
        rbm.log.to_csv('results/learn/{}/learning_{}.csv'.format(total, i))
        print('learned')

        '''
        result_train = mensurate(rbm, pedalboards_train, plugins)
        result_train.to_csv('results/4-2/{}/result_train_{}.csv'.format(total, i))
        print('Train:', result_train.mean())

        result_test = mensurate(rbm, pedalboards_test, plugins)
        result_test.to_csv('results/4-2/{}/result_test_{}.csv'.format(total, i))
        print('Test:', result_test.mean())
        '''

if __name__ == '__main__':
    process(50, verbose=False, RANDOM_STATE=None)
    process(500, verbose=False, RANDOM_STATE=None)
    process(50000, verbose=False, RANDOM_STATE=None)
