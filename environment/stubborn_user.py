import random
import numpy

from constant import *
from human_modeling_utils import utils
from .base_environment import BaseEnvironment

class StubbornUser(BaseEnvironment):
    """
    StubbornUser behaves in the following way: It distinguishes the state very clearly. For each
    state, he either always responds notifications, or always dismisses notifications.
    """

    def __init__(self):
        self.behavior = {}
        for sTime in utils.allTimeStates():
            for sDay in utils.allDayStates():
                for sNotification in utils.allLastNotificationStates():
                    state = (sTime, sDay, sNotification)
                    self.behavior[state] = (random.random() < 0.3)

    def getResponseDistribution(self, hour, minute, day, lastNotificationTime):

        stateTime = utils.getTimeState(hour, minute)
        stateDay = utils.getDayState(day)
        stateNotification = utils.getLastNotificationState(lastNotificationTime)
        state = (stateTime, stateDay, stateNotification)

        probAnswerNotification = (1.0 if self.behavior[state] else 0.0)
        probIgnoreNotification = 1.0 - probAnswerNotification
        return (probAnswerNotification, probIgnoreNotification)
