from Explainability.CompletePolicyGraph import CompletePolicyGraph
from Code.Explainability.PartialPolicyGraph import PartialPolicyGraph
from datetime import datetime
import csv
import os

class Experiment:
    '''
    Executes an experiment

    '''
    def __init__(self, id, rl_agent_id,
                 discretizer,
                 pg_algorithm,
                 layout='simple',
                 description=""):
        """ Constructor

        :param id: ID of the experiment
        :param rl_agent_id: RL agent model ID
        :param discretizer: Which Discretizer algorithm do we want to use?
        :param pg_algorithm: Which PG lgorithm do we want to use?
        :param layout: Layout to run the game
        :param description: Experiment description
        """
        self.id = id
        self.layout = layout
        self.discretizer = discretizer
        self.pg_algorithm = pg_algorithm

        self.diff_aer = None
        self.diff_std = None
        self.transferred_learning = None
        self.percentage_new_states = None
        self.percentage_seen_states = None
        self.percentage_STD = None

        self.rl_agent_id = rl_agent_id
        self.ego_file = 'Code/PantheonRL/models/ego{}'.format(self.rl_agent_id)
        self.alt_file = 'Code/PantheonRL/models/alt{}'.format(self.rl_agent_id)

        discretizer_ID = ''.join(filter(str.isdigit, discretizer.__name__))
        pg_algorithm_ID = pg_algorithm.__name__[0].upper()
        experiment_name = f"{pg_algorithm_ID}_D{discretizer_ID}"
        self.experiment_name = f'Exp_{experiment_name}'

        if not os.path.exists(f'Code/MDP_Results/{self.layout}'):
            os.makedirs(f'Code/MDP_Results/{self.layout}')
        if not os.path.exists(f'Code/MDP_Results/{self.layout}/{self.id}'):
            os.makedirs(f'Code/MDP_Results/{self.layout}/{self.id}')

        if not os.path.exists(f'Code/MDP_Models/{self.layout}'):
            os.makedirs(f'Code/MDP_Models/{self.layout}')
        if not os.path.exists(f'Code/MDP_Models/{self.layout}/{self.id}'):
            os.makedirs(f'Code/MDP_Models/{self.layout}/{self.id}')

        self.results_folder = f"Code/MDP_Results/{self.layout}/{self.id}/"
        self.models_folder = f"Code/MDP_Models/{self.layout}/{self.id}/"
        self.description = description

    def run(self, train=True, test=True,
            train_verbose=False, test_verbose=False,
            train_episodes=5, train_seeds=range(1, 100, 1),
            test_episodes=3, test_seeds=range(100, 105, 1),
            ask_questions_xai=False,
            subgraph=None):

        """ Executes an Experiment

        :param train:           True if we want to train the agent, else load it
        :param test:            True if we want ot test the agent
        :param train_verbose:   True if we want to see train_verbose
        :param test_verbose:    True if we want to see test_verbose
        :param train_episodes:  Number of training episodes
        :param train_seeds:     Number of training seeds
        :param test_episodes:   Number of test episodes
        :param test_seeds:      Number of test seeds
        :param ask_questions_xai:   True if we want to plt XAI Menu (for explanations)
        :return:
        """

        # Create the PG object
        pg = self.pg_algorithm(self.ego_file, self.alt_file, self.discretizer, name=self.experiment_name, layout=self.layout)

        self.description += "\nPredicates: \n\t{}\n".format(list(pg.discretizer.get_predicate_space().keys()))

        if train:
            # Start feeding the PG
            pg.feed(seeds=train_seeds, num_episodes=train_episodes, verbose=train_verbose)
            # Saves the model
            pg.save(self.models_folder)
        else:
            # Loads the model
            pg.load(self.models_folder)

        if test:
            # Tests fed MDP Agent.
            pg.test(seeds=test_seeds, num_episodes=test_episodes, verbose=test_verbose)
            # Compares both agents (RL vs MDP)
            self.diff_aer, self.diff_std, \
            self.transferred_learning, self.percentage_new_states, \
            self.percentage_seen_states, self.percentage_STD  = pg.compare()

            self.__save_parameters_to_file(train, test, train_verbose, test_verbose, train_episodes, train_seeds, test_episodes, test_seeds)

        # Visualize the resulting MDP Agent
        pg.show_interactive(second_display=True, show_options=False, subgraph=subgraph)

        # Questions XAI
        if ask_questions_xai:
            self.questions_xai(pg)


    def feed_and_test(self, batch_size=5,
                      test_seeds=5, test_episodes=5,
                      train_seeds=300, train_episodes=5,
                      train_verbose=False, test_verbose=False):
        """ Trains and tests and agent using batches. Also generates a files with all the results.

        :param batch_size:      Number of seeds used to train a batch
        :param train_episodes:  Number of training episodes
        :param train_seeds:     Number of training seeds
        :param test_episodes:   Number of test episodes
        :param test_seeds:      Number of test seeds
        :return:
        """

        f = open(f'{self.models_folder}{self.experiment_name}_Results.txt', 'w+')
        csv_file = open(f'{self.results_folder}{self.experiment_name}.csv', 'w+')
        csv_writer = csv.writer(csv_file)

        # Write header
        header = ['Seed Ini', 'Seed End', 'AER', 'STD', 'TL %', 'New states %', 'Seen states %', 'STD %']
        csv_writer.writerow(header)

        run_episodes = 0

        # Create the PG object
        pg = self.pg_algorithm(self.ego_file, self.alt_file, self.discretizer, name=self.experiment_name,
                               layout=self.layout)
        self.description += "\nPredicates: \n\t{}\n".format(list(pg.discretizer.get_predicate_space().keys()))

        try:
            f.write('Description:\n')
            f.write('----------------\n')
            f.write(self.description + '\n')
            f.write('----------------\n\n')
            f.write(f"Training Episodes: {train_episodes}, Test Episodes: {test_episodes}, PG Algorithm: {self.pg_algorithm}, Discretizer: {self.discretizer}\n")
            for seed in range(0, train_seeds, batch_size):
                print('+++++++++++++++++++++++ SEED', seed)
                # Feed PG with one seed
                pg.feed(seeds=range(seed, seed + batch_size), num_episodes=train_episodes, verbose=train_verbose)

                # Tests fed MDP Agent.
                pg.test(seeds=range(1000+seed, 1000+seed + test_seeds), num_episodes=test_episodes, verbose=test_verbose)

                run_episodes += train_episodes
                # Compares both agents (RL vs MDP)
                self.diff_aer, self.diff_std, \
                self.transferred_learning, self.percentage_new_states, \
                self.percentage_seen_states, self.percentage_STD = pg.compare()

                data = [seed,
                        seed + batch_size,
                        round(self.diff_aer, 1),
                        round(self.diff_std, 1),
                        self.transferred_learning,
                        round(self.percentage_new_states * 100, 2),
                        round(self.percentage_seen_states * 100, 4),
                        round(self.percentage_STD * 100, 4)
                        ]
                f.write("Seed: {}-{}\tAER: {}\tSTD: {}\tTransfered Learning: {}%"
                        "\tNew States: {}%\tSeen States: {}%\tSTD: {}%\n".format(*data))
                csv_writer.writerow(data)

                # Saves the model
                pg.save(self.models_folder)
                self.__save_parameters_to_file(True, True, train_verbose, test_verbose, train_episodes, train_seeds, test_episodes, test_seeds)

                # Visualize the resulting MDP Agent
                pg.show_interactive()

            f.close()
            csv_file.close()

        except KeyboardInterrupt:
            # Saves the model
            pg.save(self.models_folder)
            self.__save_parameters_to_file(True, True, train_verbose, test_verbose, train_episodes, train_seeds, test_episodes, test_seeds)

    def questions_xai(self, pg):
        """
        Plots the explainability menu.
        :param pg: Policy Graph
        :return:
        """
        while True:
            print('* What do you want to ask?')
            print('\t1: What will you do when you are in state __?')
            print('\t2: When do you perform action __?')
            print('\t3: Why do you perform action __ in state __?')
            print('\t4: Exit')
            option = int(input('Option: '))
            if option == 1:
                # Question 1: What will you do when you are in state X?
                pg.question1()
            elif option == 2:
                # Question 1: What will you do when you are in state X?
                pg.question2()
            elif option == 3:
                # Question 1: What will you do when you are in state X?
                pg.question3()
            else:
                break
            print('\n---------------------------------')

    def __save_parameters_to_file(self, train, test, train_verbose, test_verbose, train_episodes, train_seeds, test_episodes, test_seeds):
        """ Saves all the used parameters in a file
        :return:
        """
        f = open(f'{self.models_folder}{self.experiment_name}_parameters.txt', 'w+')
        text = f"""
Description:
----------------
{self.description}

Parameters used:
----------------

\t- Date: \t{datetime.now()},
\t- ID: \t{self.id},
\t- RL Agent ID: {self.rl_agent_id},
\t- Discretizer: {self.discretizer},
\t- MDP Algorithm: {self.pg_algorithm},
\t- Train: {train},
\t- Test: {test},
\t- Train Verbose: {train_verbose},
\t- Test Verbose: {test_verbose},
\t- Layout: {self.layout},
\t- Train Episodes: {train_episodes},
\t- Train Seeds: {train_seeds},
\t- Test Episodes: {test_episodes},
\t- Test Seeds: {test_seeds}

Results:
---------

"""
        if self.diff_aer is not None:
            text += '\t- Average Episode Reward Diff: ' + str(self.diff_aer) + '\n'
        if self.diff_std is not None:
            text += '\t- Standard Deviation Diff: ' + str(self.diff_std) + '\n'
        if self.transferred_learning is not None:
            text += '\t- Transfered Learning: ' + str(self.transferred_learning) + '%\n'
        if self.percentage_new_states is not None:
            text += '\t- New states: ' + str(self.percentage_new_states * 100) + '%\n'
        if self.percentage_seen_states is not None:
            text += '\t- Seen states: ' + str(self.percentage_seen_states * 100) + '%\n'
        if self.percentage_STD is not None:
            text += '\t- STD: ' + str(self.percentage_STD * 100) + '%\n'
        f.write(text)
        f.close()