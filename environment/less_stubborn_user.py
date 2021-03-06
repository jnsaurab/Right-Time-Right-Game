  
import random
import numpy

from constant import *
from human_modeling_utils import utils
from .base_environment import BaseEnvironment

class LessStubbornUser(BaseEnvironment):
    """
    LessStubbornUser behaves similarly to Stubborn, the difference is that instead of being
    deterministic on each state, a stochastic process is introduced: It has a certain probability
    to deviate away from his original plan.
    """

    def __init__(self, deviationProb=0.1):
        self.behavior = {}
        for sTime in utils.allTimeStates():
            for sDay in utils.allDayStates():
                for sNotification in utils.allLastNotificationStates():
                    state = (sTime, sDay, sNotification)
                    self.behavior[state] = (random.random() < 0.5)
        self.probTake = 1. - deviationProb
        self.probNotTake = deviationProb

    def getResponseDistribution(self, hour, minute, day, lastNotificationTime):
        stateTime = utils.getTimeState(hour, minute)
        stateDay = utils.getDayState(day)
        stateNotification = utils.getLastNotificationState(lastNotificationTime)
        state = (stateTime, stateDay, stateNotification)

        probAnswerNotification = (self.probTake if self.behavior[state] else self.probNotTake)
        probIgnoreNotification = 1.0 - probAnswerNotification
      
        return (probAnswerNotification, probIgnoreNotification)
