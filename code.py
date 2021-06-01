import copy
import random
import pandas as pd
import numpy as np
import time
np.random.seed(13)
random.seed(13)
df = pd.read_csv('truth_table.csv')
rows, columns = df.shape
inputs = columns - 1
gates = inputs - 1

population = 1024

people = []
for i in range(population):
    chromosome = []
    for j in range(gates):
        chromosome.append(random.randint(0, 5))
    people.append(chromosome)
people = np.array(people)

def fitness(chromosome):
    col = df.iloc[:, 0].to_numpy(dtype = bool, copy = True)
    for i in range(gates):
        if chromosome[i] == 0:
            np.logical_and(col, df.iloc[:, i + 1].to_numpy(dtype = bool, copy = True), out = col)
        elif chromosome[i] == 1:
            np.logical_or(col, df.iloc[:, i + 1].to_numpy(dtype = bool, copy = True), out = col)
        elif chromosome[i] == 2:
            np.logical_xor(col, df.iloc[:, i + 1].to_numpy(dtype = bool, copy = True), out = col)
        elif chromosome[i] == 3:
            np.logical_not(np.logical_and(col, df.iloc[:, i + 1].to_numpy(dtype = bool, copy = True)), out = col)
        elif chromosome[i] == 4:
            np.logical_not(np.logical_or(col, df.iloc[:, i + 1].to_numpy(dtype = bool, copy = True)), out = col)
        else:
            np.logical_not(np.logical_xor(col, df.iloc[:, i + 1].to_numpy(dtype = bool, copy = True)), out = col)
    return np.equal(col, df.iloc[:, columns - 1]).sum()

def populationMax(people):
    return max(map(fitness, people))

def populationFitness(people):
    return sum(map(fitness, people)) / population

prob = [2 * i / (population * (population + 1)) for i in range(1, population + 1)]

def FPS(people):
    s = 0
    for person in people:
        s += fitness(person)
    index = np.random.choice(ind, population, p = [fitness(person) / s for person in people])
    print(index)
    return [people[i] for i in index]

resh = np.arange(population).reshape(-1, 1)
def rankBasedSelection(people):
    fitnessList = np.array(list(map(fitness, people)))
    pairList = np.hstack((fitnessList.reshape(-1,1), resh))
    sortedList = pairList[pairList[:, 0].argsort()]
    index = np.random.choice(population, population, p = prob)
    newList = sortedList[index]
    return np.array(people)[(newList[:, 1]).astype(int)]

def onePointCrossover(people, probability):
    for i in range(population // 2):
        if random.random() <= probability:
            index = random.randint(1, gates - 1)
            for j in range(index):
                people[2 * i][j], people[2 * i + 1][j] = people[2 * i + 1][j], people[2 * i][j]

def mutation(people, probability):
    for i in range(population):
        for j in range(gates):
            if random.random() <= probability:
                people[i][j] = random.randint(0, 5)

def hasSolution(people):
    return max(map(fitness, people)) == rows

def printSolution(chromosome):
    ans = ""
    for gene in chromosome:
        if gene == 0:
            ans += "AND "
        elif gene == 1:
            ans += "OR "
        elif gene == 2:
            ans += "XOR "
        elif gene == 3:
            ans += "NAND "
        elif gene == 4:
            ans += "NOR "
        else:
            ans += "XNOR "
    return ans

while True:
    print("Max fitness: ", populationMax(people))
    people = rankBasedSelection(people)
    random.shuffle(people)
    onePointCrossover(people, 0.7)
    mutation(people, 0.01)
    if hasSolution(people):
        break

for person in people:
    if fitness(person) == rows:
        print(printSolution(person))
        break

