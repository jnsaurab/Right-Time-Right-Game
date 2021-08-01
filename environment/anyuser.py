import sys
import csv
import numpy as np
from collections import Counter

from constant import *
from human_modeling_utils import utils
from .base_environment import BaseEnvironment


class AnySurveyUser(BaseEnvironment):
    """
    The idea here is that for any given data we process data and collect the records regarding it.
    The user's mental status is determined based on the following strategy: Given the time , day,
    and last notification response time, I filter out the relevant records. I assign a weight
    for each record based on the inverse of the time delta.
    """

    def __init__(self, filePaths, filterFunc=None, dismissWarningMsg=False):
        # self.behavior is a dictionary of lists. The values
        # are the relavent records
        self.behavior = {}
        for sTime in utils.allTimeStates():
            for sDay in utils.allDayStates():
                for sNotification in utils.allLastNotificationStates():
                    state = (sTime, sDay,sNotification)
                    self.behavior[state] = []

        self.records = []
        for filePath in filePaths:
            self.records.extend(self._parseFile(filePath))

        # apply the filter
        if filterFunc:
            self.records = list(filter(filterFunc, self.records))

        # arrange the records to the correct category in self.behavior
        for r in self.records:
            state = (r['stateTime'], r['stateDay'],r['stateNotification'])
            self.behavior[state].append(r)

        self.numNoDataStates = 0
        for state in self.behavior:
            if len(self.behavior[state]) == 0:
                sTime, sDay, sNotification = state
                if not dismissWarningMsg:
                    sys.stderr.write("No record for day=%d, location=%d, activity=%d, notification=%d\n"
                            % (sDay, sLocation, sActivity, sNotification))
                self.numNoDataStates += 1

        # display warning message
        if not dismissWarningMsg:
            if self.numNoDataStates > 0:
                sys.stderr.write("WARNING: No records for %d states. The behavior will be random.\n"
                        % self.numNoDataStates)

    def getResponseDistribution(self, hour, minute, day,
            lastNotificationTime):
        stateTime = utils.getTimeState(hour, day)
        stateDay = utils.getDayState(day)
        stateNotification = utils.getLastNotificationState(lastNotificationTime)
        state = (stateTime ,stateDay, stateNotification)

        records = self.behavior[state]
        
        if len(records) == 0:
            probAnswerNotification = 0.1
            probIgnoreNotification = 0.8
            probDismissNotification = 0.1
        else:
            timeDiffs = [abs(utils.getDeltaMinutes(0, hour, minute, 0, r['rawHour'], r['rawMinute']))
                    for r in records]
            weights = np.array([(1. / (t + 5.)) ** 1 for t in timeDiffs])
            weightSum = np.sum(weights)
            probs = weights / weightSum

            chosenRecord = np.random.choice(a=records, p=probs)

            probAnswerNotification, probIgnoreNotification = 0.0, 0.0, 0.0
            if chosenRecord['answerNotification'] == ANSWER_NOTIFICATION_ACCEPT:
                probAnswerNotification = 1.0
            elif chosenRecord['answerNotification'] == ANSWER_NOTIFICATION_IGNORE:
                probIgnoreNotification = 1.0
            

        return (probAnswerNotification, probIgnoreNotification)

    def getNumTotalRecords(self):
        return len(self.records)

    def getNumNoDataStates(self):
        return self.numNoDataStates

    def getNumRecordsAcceptingNotification(self):
        return len([r for r in self.records
                if r['answerNotification'] == ANSWER_NOTIFICATION_ACCEPT])

    def getNumRecordsIgnoringNotification(self):
        return len([r for r in self.records
                if r['answerNotification'] == ANSWER_NOTIFICATION_IGNORE])


    def _parseFile(self, filename):
        """
        This function receives a csv file obtained from mTurk and convert it to a list of
        dictionary objects. Please see `_parseLine()` for the format of the dictionary object.
        """
        with open(filename) as f:
            reader = csv.DictReader(f)
            records = [self._parseCsvRow(row) for row in reader]
        
        return [r for r in records if r is not None]

    def _parseCsvRow(self, row):
        """
        This function receives a line from the input file and convert it to a dictionary with
        the following keys:
            parsedRow, rawHour, rawMinute,
            rawWorkerID, rawWorkingTimeSec,
            stateDay, stateLocation, stateActivity, stateNotification,
            answerNotification
        If the line is not able to converted, or the response is invalid, `None` is returned
        instead.
        """
        hour = int(row['Input.hour'])
        minute = int(row['Input.minute'])
        day = int(row['Input.day'])
        lastSeenNotificationTime = int(row['Input.last_notification_time'])
        response = row['Answer.sentiment']

        answerNotificationCriteria = {
            'Dismiss': None,
            'Accept': ANSWER_NOTIFICATION_ACCEPT,
            'Later': ANSWER_NOTIFICATION_IGNORE,
            'Invalid': None,
        }
        answerNotification = answerNotificationCriteria[response]
        if answerNotification is None:
            return None

        return {
            'parsedRow': row,
            'rawHour': hour,
            'rawMinute': minute,
            'stateTime' : utils.getTimeState(hour, minute),
            'stateDay': utils.getDayState(day),
            'stateNotification': utils.getLastNotificationState(lastSeenNotificationTime),
            'answerNotification': answerNotification,
        }
