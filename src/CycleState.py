#!/usr/bin/env python

"""
Author: David Quail, January, 2018.
Description:
Creates a state to be placed into the cycle world.

"""

class CycleState:

    def __init__(self):
        self.isCreated = True
        self.color = "w"
        self.isTerminal = False