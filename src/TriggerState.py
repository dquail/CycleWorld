#!/usr/bin/env python

"""
Author: David Quail, February, 2018.
Description:
Creates a state to be placed into the trigger world.

"""

class TriggerState:

    def __init__(self):
        self.isCreated = True
        self.color = "w"
        self.triggerIsTerminal = False