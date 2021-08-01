import random
import numpy as np

from constant import *
from human_modeling_utils import utils
from human_modeling_utils.chronometer import Chronometer


class Controller:

    def __init__(self, agent, environment,
            simulationWeek=10, verbose=True):
        self.rewardCriteria = {
                ANSWER_NOTIFICATION_ACCEPT: 3,
                ANSWER_NOTIFICATION_IGNORE: -1,
        }

        self.verbose = verbose

        # set chronometer which automatically skips 10pm to 8am because it's usually when people
        # sleep
        self.chronometer = Chronometer(skipFunc=(lambda hour, _m, _d: hour < 8 or hour >= 23))

        self.stepWidthMinutes = 10
        self.simulationWeek = simulationWeek

        self.lastNotificationMinute = 0
        self.lastNotificationHour = 0
        self.lastNotificationNumDays = 0

        self.agent = agent
        self.environment = environment

        self.simulationResults = []


    def execute(self):
        numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.forward(
                self.stepWidthMinutes)

        while numDaysPassed < self.simulationWeek * 7:
            if self.verbose:
                print("Day %d %d:%02d" % (numDaysPassed, currentHour, currentMinute))

            # get environment info (user context)
            lastNotificationTime = utils.getDeltaMinutes(
                    numDaysPassed, currentHour, currentMinute,
                    self.lastNotificationNumDays, self.lastNotificationHour, self.lastNotificationMinute,
            )
            stateLastNotification = utils.getLastNotificationState(lastNotificationTime)

            probAnsweringNotification, probIgnoringNotification = (
                    self.environment.getResponseDistribution(
                        currentHour, currentMinute, currentDay,
                        lastNotificationTime,
                    )
            )
            probAnsweringNotification, probIgnoringNotification = utils.normalize(
                    probAnsweringNotification, probIgnoringNotification)

            # prepare observables and get action
            stateTime = utils.getTimeState(currentHour, currentMinute)
            stateDay = utils.getDayState(currentDay)
            sendNotification = self.agent.getAction(stateTime, stateDay, stateLastNotification)

            # calculate reward
            if not sendNotification:
                reward = 0
            else:
                userReaction = np.random.choice(
                        a=[ANSWER_NOTIFICATION_ACCEPT, ANSWER_NOTIFICATION_IGNORE],
                        p=[probAnsweringNotification, probIgnoringNotification],
                )
                reward = self.rewardCriteria[userReaction]
                self.lastNotificationNumDays = numDaysPassed
                self.lastNotificationHour = currentHour
                self.lastNotificationMinute = currentMinute
            self.agent.feedReward(reward)

            # log this session
            self.simulationResults.append({
                    'context': {
                        'numDaysPassed': numDaysPassed,
                        'hour': currentHour,
                        'minute': currentMinute,
                        'day': currentDay,
                        'lastNotification': lastNotificationTime,
                    },
                    'probOfAnswering': probAnsweringNotification,
                    'probOfIgnoring': probIgnoringNotification,
                    'decision': sendNotification,
                    'reward': reward,
            })

            # get the next decision time point
            numDaysPassed, currentHour, currentMinute, currentDay = self.chronometer.forward(
                    self.stepWidthMinutes)

        return self.simulationResults
