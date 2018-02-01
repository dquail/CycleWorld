from random import randint

class BehaviorPolicy:
    """
    'L'= gp left
    'R' = go right
    """
    def __init__(self):
        self.lastAction = 0
        self.i = 0

    def policy(self, state):
        self.i = self.i + 1
        #return self.randomPolicy(state)
        return self.moveLeftPolicy(state)

    def randomPolicy(self, state):
        actions = ["L","L", "L", "L", "L", "R", "R", "R", "R", "R"]
        action = actions[randint(0,9)]
        return action

    def moveLeftPolicy(self, state):
        self.lastAction = "L"
        return "L"

    def epsilonGreedyPolicy(self, state):
        print("Do something here")
