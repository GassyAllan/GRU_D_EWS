import pandas as pd

'''
Functions to calculate MEWS based on MEWS parameters. https://www.mdcalc.com/calc/1875/modified-early-warning-score-mews-clinical-deterioration
'''

def mews_temp (temp):
    if temp == 0:
        return 0
    elif temp <= 35.0:
        return 2
    elif temp >= 38.5:
        return 2
    else:
        return 0

def mews_resp (resp):
    if resp == 0:
        return 0
    elif resp >= 30:
        return 3
    elif resp >= 21:
        return 2
    elif resp >= 15:
        return 1
    elif resp < 9:
        return 2
    else:
        return 0

def mews_hr (hr):
    if hr == 0:
        return 0
    elif hr >= 130:
        return 3
    elif hr >= 111 or hr <= 40 :
        return 2
    elif hr >= 51 and hr <= 90:
        return 0
    else:
        return 1

#Systolic BP only taken into account
def mews_bp (bp):
    if bp == 0:
        return 0
    elif bp >= 200:
        return 2
    elif bp <= 70:
        return 3
    elif bp <=80:
        return 2
    elif bp <=100:
        return 1
    else:
        return 0

# Takes series from DF to extract values to then calculate mews based on functions above 
def mews_calc (x):
    mews = 0
    mews += mews_temp((x['temp']))
    mews += mews_hr(x['hr'])
    mews += mews_resp(x['rr'])
    mews += mews_bp(x['sbp'])
    return mews

'''
Functions to calculate CART based on . https://www.mdcalc.com/calc/10029/cart-cardiac-arrest-risk-triage-score
'''

def cart_age (age):
    if age >= 70:
        return 9
    elif age >= 55:
        return 4
    else:
        return 0

def cart_resp (resp):
    if resp >= 30:
        return 22
    elif resp <= 20:
        return 0
    elif resp <= 23:
        return 8
    elif resp <= 25:
        return 12  
    else:
        return 15

def cart_hr (hr):
    if hr >= 140:
        return 13
    elif hr <= 109:
        return 0
    else:
        return 4   

def cart_dbp (bp):
    if bp >= 50:
        return 0
    elif bp <= 34:
        return 13
    elif bp <= 39:
        return 6
    else:
        return 4

# Takes series from DF to extract values to then calculate NEWS based on functions above 
def cart_calc (x):
    cart = 0
    cart += cart_hr(x['hr'])
    cart += cart_resp(x['rr'])
    cart += cart_age(x['age'])
    cart += cart_dbp(x['dbp'])
    return cart

'''
Functions to calculate NEWS based on NEWS2 parameters. https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2
'''

def news_temp (temp):
    if temp == 0:
        return 0
    elif temp <= 35.0:
        return 3
    elif temp >= 39.1:
        return 2
    elif temp >= 36.1 and temp <=39.0:
        return 0
    else:
        return 1

def news_resp (resp):
    if resp == 0:
        return 0
    elif resp <= 8 or resp >= 25:
        return 3
    elif resp >= 21:
        return 2
    elif resp <= 11:
        return 1
    else:
        return 0

def news_hr (hr):
    if hr == 0:
        return 0
    elif hr <= 40 or hr >= 131:
        return 3
    elif hr >= 111:
        return 2
    elif hr >= 51 and hr <= 90:
        return 0
    else:
        return 1

# Assumes not on oxygen, May need to consider how to account for this in the overall calculator
def news_spo2 (spo2, flowrate):
    # Check if on O2
    if flowrate == 0:
        if spo2 == 0:
            return 0
        elif spo2 <= 91 :
            return 3
        elif spo2 <= 93:
            return 2
        elif spo2 <= 95:
            return 1
        else:
            return 0
    else:
        # Being on supplemental O2 + 2
        if spo2 == 0:
            return 0
        elif spo2 <= 83 or spo2 >= 97:
            return 5
        elif spo2 <= 85 or spo2 >= 95:
            return 4
        elif spo2 <= 95 or spo2 >= 93:
            return 3
        else:
            return 0

#Systolic BP only taken into account
def news_bp (bp):
    if bp == 0:
        return 0
    elif bp <= 90 or bp >= 220:
        return 3
    elif bp <= 100:
        return 2
    elif bp <= 110:
        return 1
    else:
        return 0

# Takes series from DF to extract values to then calculate NEWS based on functions above 
def news_calc (x):
    news = 0
    news += (news_temp((x['temp'])))
    news += (news_hr(x['hr']))
    news += (news_resp(x['rr']))
    news += (news_spo2(x['spo2'], x['flow']))
    news += (news_bp(x['sbp']))
    return news
