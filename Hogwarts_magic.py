#!/usr/bin/env python
# coding: utf-8

import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def get_data():
    data = pd.read_csv('dataset_train.csv')
    data = data.fillna(0)
    return data


def train_data(data):
    train = data.drop(labels='Index',axis=1)
    x_train = train.iloc[:,1:]
    y_train = train.iloc[:, 0]
    return x_train, y_train


def to_cat(strs):
    Alpha = {i:j for j,i in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    add = 0
    for s in strs:
        add+=Alpha[s.upper()]
    return round(add/len(strs), 2)


data = get_data()
x_train, y_train = train_data(data)

encoding = {i:j for j,i in enumerate(y_train.unique())}
def encode(strs):
    return encoding[strs]


def clean_prepare(x_train):
    df = pd.get_dummies(x_train["Best Hand"], prefix="Hand")
    x_train = x_train.join(df)
    x_train["Birthday"] = pd.to_datetime(x_train["Birthday"])

    start =  pd.Timestamp("1970-01-01")
    print(start)

    x_train["Birthday"] = (x_train["Birthday"] - start)//pd.Timedelta('1s')
    x_train = x_train.drop(labels='Best Hand',axis = 1)
    x_train["First Name"] = x_train["First Name"].apply(to_cat)
    x_train["Last Name"] = x_train["Last Name"].apply(to_cat)

    print(x_train.head())
    return x_train

def trained_model(x_train, y_train, data):
    y_train = y_train.apply(encode)
    forest = RandomForestClassifier(n_estimators=200,n_jobs=-1)
    y_trains = pd.get_dummies(data.iloc[:, 1])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_trains,
                                                  test_size=0.1,random_state=42)
    forest.fit(x_train, y_train)
    return forest, x_train, y_train, x_val, y_val


def get_prepare_test():
    d_test = pd.read_csv('dataset_test.csv')

    start = pd.Timestamp('1970-01-01')

    test = d_test.drop(labels='Index',axis=1)
    test = test.drop(labels='Hogwarts House', axis=1)

    df = pd.get_dummies(d_test["Best Hand"], prefix="Hand")

    test = test.join(df)
    test = test.drop(labels='Best Hand',axis=1)
    test["Birthday"] = pd.to_datetime(test["Birthday"])
    test["First Name"] = test["First Name"].apply(to_cat)
    test["Last Name"] = test["Last Name"].apply(to_cat)
    test = test.fillna(0)
    test["Birthday"] = (test["Birthday"] - start)//pd.Timedelta('1s')

    print(test.head())
    return test

def get_test_results(test, forest, y_train):
    y_test = forest.predict(test)
    encodes = []
    for i in y_test:
        for j in range(len(y_train.columns)):
            if i[j] == 1:
                encodes.append(y_train.columns[j])
    test["House"] = encodes
    print(test.head())
    return test

def  get_prediction(get_result):

x_train = clean_prepare(x_train)
forest, x_train, y_train, x_val, y_val = trained_model(x_train, y_train, data)

test = get_prepare_test()
test = get_test_results(test, forest, y_train)

print(classification_report(y_val,forest.predict(x_val),
                            target_names=y_train.columns))
