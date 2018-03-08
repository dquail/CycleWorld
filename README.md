# CycleWorld
Predictive feature learning for a reinforcement learning task.

## Abstract
In some reinforcement learning environments, direct observations lack sufficient information to directly represent the agents state. One approach, in an effort to suffienctly describe agent state, is to augment these observations with predictive features. The experiments in this repository are conducted in such an information lacking environment, and attempt to discover useful predictive features.


## Introduction
A "feature" of an agents state can be considered as some measurable quantity of the current state it is in. As a human agent, this may be something like the color I am looking at, the room I am standing in, or whether I'm hot or cold. The most obvious features are those that describe the agents immediate observations, like the ones just mentioned. Other abstract features may be considered however. Those which encapsulate the past, such as which action it just took or how many steps it has taken forward.
A second type of more abstract feature would be those which predict some future quantity. For example, consider a basketball player standing at middle court. It may describe it's current state by a series of action based predictions.
- If I were to pass the ball to my teammate, what is the probability they catch it?
- If I were to shoot, what is the probability it will go in the net?
- If I were to continue dribbling forward, how long would it be until a turnover occured?
These predictive features are, in many ways, similar to historical features. Instead of looking to the past to describe the present, you are looking toward what is likely to happen in the future, to describe the present.
Historical, and predictive features are good for genaralizing states. ie. Regardless of all other factors (where the player and her teammates are on the court, and the color of her shoes), it may be significant to know what is the probability she scores if she shoots?
Another strength of these types of features is demonstrated in environments where information is lacking in current observations. To best describe this, consider an example.

Imagine a human agent standing in the middle of a football field during a severe snow storm. Every direction the agent looks, appears to be the same - all they see is "white." If the agent were to only represent it's current state by considering these immediate observations, it would be lost, with no hope for navigating to the goal post. However, consider the agent constructing it's state differently. Imagine if the agent were to create features such as:
- If I were to walk straight, would I eventually end up in the oppositions endzone?
- If I were to turn left and then walk straight, would I eventually get indoors?
- If I were to walk straight, how many steps would it take to hit a wall?

If the agent constructs, and estimates the right set of predictions, it's able identify where it is in the field, despite "only seeing white"?

It is such a setting that we look to create an agent architecture that is able to learn such predictive features that will allow the agent to represent it's state.

## Environment
Our environment is a slight adaptation to the Cycle world. The MDP is illustrated below.

![alt text](Documentation/WriteUp/Images/SegmentExample2.png "Segment 2 example")

There are 6 states oriented in a circle. Each state leads to the next in a clockwise fashion. The differnce is that we introduce a second action we call "trigger." In all but one of the states, "trigger" transitions the state back to itself. In the one special state, tacking the "trigger" action takes the agent to a terminal state. The reward for each episode is -1. It is easy to see, that for such an environment, the optimal policy is one which moves forward at each state - except the "special state" - whose optimal action is to pull the trigger. The difficulty of learning this policy is that there is a lot of state aliasing. Each "white" state looks exactly the same. Which action should be taken? With such aliasing, the agent will either always move, or always pull the trigger, when it is in a white state. Clearly, neither option results in an optimal policy.

## Predictive state representation to the rescue ...

