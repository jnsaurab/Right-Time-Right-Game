from constant import *

def getTimeState(hour, minute):
    if hour < 8:
        return STATE_TIME_SLEEPING
    elif hour < 12:
        return STATE_TIME_MORNING
    elif hour < 18:
        return STATE_TIME_AFTERNOON
    else:
        return STATE_TIME_EVENING

def getDayState(day):
    if day == 0:
        return STATE_DAY_MONDAY
    elif day == 1:
        return STATE_DAY_TUESDAY
    elif day == 2:
        return STATE_DAY_WEDNESDAY
    elif day == 3:
        return STATE_DAY_THURSDAY
    elif day == 4:
        return STATE_DAY_FRIDAY
    elif day == 5:
        return STATE_DAY_SATURDAY
    else:
        return STATE_DAY_SUNDAY

def getLastNotificationState(last_notification_time):
    if last_notification_time <= 60:
        return STATE_LAST_NOTIFICATION_WITHIN_1HR
    else:
        return STATE_LAST_NOTIFICATION_LONG

def getDeltaMinutes(day1, hour1, minute1, day2, hour2, minute2):
    return (day1 - day2) * 24 * 60 + (hour1 - hour2) * 60 + (minute1 - minute2)

def allTimeStates():
    return [STATE_TIME_MORNING, STATE_TIME_AFTERNOON, STATE_TIME_EVENING, STATE_TIME_SLEEPING]

def allDayStates():
    return [STATE_DAY_MONDAY, STATE_DAY_TUESDAY, STATE_DAY_WEDNESDAY, STATE_DAY_THURSDAY, STATE_DAY_FRIDAY, STATE_DAY_SATURDAY, STATE_DAY_SUNDAY]

def allLastNotificationStates():
    return [STATE_LAST_NOTIFICATION_WITHIN_1HR, STATE_LAST_NOTIFICATION_LONG]

def normalize(*args):
    valSum = sum(args)
    return [v / valSum for v in args]

def argmaxDict(d):
    idx = None
    val = -1e100
    for k in d:
        if d[k] > val:
            idx, val = k, d[k]
    return idx

def maxDictVal(d):
    return max([d[k] for k in d])

def clip(val, min_cut, max_cut):
    return min(max(val, min_cut), max_cut)
