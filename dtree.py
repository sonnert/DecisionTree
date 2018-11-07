import numpy as np
import pandas as pd
import pprint

"""
TLDR

Entropy = Unpredictability

Create tree recursively where the feature to split on is decided by which has the lowest entropy (highest information). 
Stop splitting when all nodes pure, no pruning.
"""

class Dtree:
    def __init__(self, df, target, features=None, max_depth=5):
        self.df = df
        self.target = target
        self.max_depth = max_depth

        if features is None:
            self.features = df.columns.drop(self.target)
        else:
            self.features = features

    def get_entropy_df(self, df=None):
        """
        Returns the entropy from the entire DataFrame to the given target.
        """
        if df is None:
            df = self.df
        target = self.target

        entropy = 0
        values = df[target].unique()

        for value in values:
            # Fraction of values of 'value' in target feature
            fraction = df[target].value_counts()[value]/len(df[target])
            entropy += -fraction*np.log2(fraction)

        return entropy

    def get_entropy_feature(self, feature, df=None):
        """
        Returns the entropy from the given feature to the given target.
        """
        if df is None:
            df = self.df
        target = self.target

        target_variables = df[target].unique()
        variables = df[feature].unique()
        entropy = 0

        # Aggregate entropy for each unique value in 'feature' feature on each unique value in target feature
        for variable in variables:
            entropy_inner = 0
            for target_variable in target_variables:
                # Number of values of 'variable' in 'feature' feature that matches current target value
                num = len(df[feature][df[feature] == variable][df[target] == target_variable])
                # Number of values of 'variable' in 'feature' feature
                den = len(df[feature][df[feature] == variable])
                # Machine epsilon
                eps = np.finfo(np.float).eps
                fraction_inner = num/(den+eps)
                entropy_inner += -fraction_inner*np.log(fraction_inner+eps)
            fraction = den/len(df)
            entropy += -fraction*entropy_inner

        return abs(entropy)

    def get_lowest_entropy_feature(self, df=None):
        """
        Returns the feature with the lowest entropy given a target variable.
        """
        if df is None:
            df = self.df
        target = self.target

        entropies = []

        for feature in self.features:
            entropies.append(self.get_entropy_df(df=df) - self.get_entropy_feature(feature=feature, df=df))

        # Quit growing if no information gain is possible locally (works 100% in 99% of cases)
        if len(set(entropies)) is 1:
            return None
        return df.keys().drop(target)[np.argmax(entropies)]

    def build_tree(self, df=None, tree=None, depth=0):
        """
        Returns a recursively built tree using entropy for determining splits.
        """
        if df is None:
            df = self.df
        target = self.target

        node = self.get_lowest_entropy_feature(df)
        if not node:
            print("Pure solution not possible in current branch...")
            return tree
        variables = df[node].unique()

        if tree is None:                    
            tree = {}
            tree[node] = {}

        for value in variables:
            subtable = df[df[node] == value].reset_index(drop=True)
            inner_variables, counts = np.unique(subtable[target], return_counts=True)                        
            
            if len(counts) == 1:
                tree[node][value] = inner_variables[0]  
            elif depth >= self.max_depth:
                return tree                  
            else:
                depth += 1        
                tree[node][value] = self.build_tree(df=subtable, depth=depth)
                   
        return tree

if __name__ == "__main__":
    """
    Builds and prints tree of given .csv and name of target variable.
    """
    DF = "NBA_player_of_the_week.csv"
    TARGET = 'Age'
    FEATURES = pd.Series(['Conference', 'Draft Year', 'Height', 'Position', 'Season', 'Seasons in league', 'Team', 'Weight'])
    MAX_DEPTH = 5

    df = pd.read_csv(DF)
    dt = Dtree(df=df, target=TARGET, features=FEATURES, max_depth=MAX_DEPTH)

    print("")
    print("Features:", dt.features.values)
    print("Target: '"+TARGET+"'")
    print("")
    print("Entropy for DataFrame:", dt.get_entropy_df())
    for feature in dt.features:
        print("Entropy for '"+feature+"':", dt.get_entropy_feature(feature=feature))
    print("")

    tree = dt.build_tree()
    print("")
    print("Resulting tree (max depth "+str(MAX_DEPTH)+")")
    pprint.pprint(tree)
    print("")