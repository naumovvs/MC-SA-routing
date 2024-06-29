import pandas as pd
from math import isnan
import numpy as np
from SA import *

class Point:

    def __init__(self, id=0):
        self.__id = id
        self.location = [0, 0]
        self.capacity = 0
        self.dead_stock = 0
        self.max_level = 0
        self.name = "Point"

        self.demands = []

    @property
    def ID(self):
        return self.__id

    @property
    def X(self) -> float:
        return self.location[0]

    @property
    def Y(self) -> float:
        return self.location[1]

    def __str__(self):
        return '%s %s (%f, %f): [%d * %f]' % (self.ID, self.name, self.X, self.Y,
                                              self.capacity, self.max_level)


class DemandData:

    def __init__(self):
        self.depot = 0
        self.points = []

        self.distance_matrix = {}

        self.sets = [] # sets of points per day
        self.usets = [] # sets of urgent deliveries per day
        self.capacity = 0 # vehicle capacity
        self.nroutes = [] # number of routes per day

    def get_point(self, _id) -> Point:
        for point in self.points:
            if point.ID == _id:
                return point
        return None

    def read_points(self, xlsx_file, verbose=False):
        df = pd.read_excel(xlsx_file)
        unique_IDs = []
        for _, row in df.iterrows():
            _id = row['id_lokalizacji']
            _id = None if isnan(_id) else int(_id)
            if _id is not None and _id not in unique_IDs:
                point = Point(id=_id)
                point.name = row['Nazwa']
                point.location = [float(row['Szer.geogr']), float(row['Dł.geogr.'])]
                capacity = row['Pojemność zb. nr 1 [dm3]']
                capacity = 0 if isnan(capacity) else int(capacity)
                point.capacity = capacity
                max_level = row['Max.poziom napełnienia [%]']
                max_level = 100 if type(max_level) is str or isnan(max_level) else float(max_level)
                point.max_level = 0.01 * max_level
                dead_stock = row['Stan martwy zb. nr 1 [dm3]']
                dead_stock = 0 if type(dead_stock) is str or isnan(dead_stock) else float(dead_stock)
                point.dead_stock = dead_stock
                self.points.append(point)
                unique_IDs.append(_id)
                if verbose:
                    print(point)

    def read_demand(self, xlsx_file, verbose=False):
        df = pd.read_excel(xlsx_file)
        for _, row in df.iterrows():
            _id = row['id_lokalizacji']
            point = self.get_point(_id)
            if point is not None:
                _period = row['id_okresu']
                point.demands[_period] = (row['stan'], row['dst'])
            else:
                print("No point with ID={}".format(_id))
        # show demand
        if verbose:
            for point in self.points:
                if len(list(point.demands.keys())) > 0:
                    res = str(point.ID)
                    for d in point.demands:
                        res += " {}({},{}) ".format(d, point.demands[d][0], point.demands[d][1])
                    print(res)

    def check_demand(self, verbose=False):
        # check if the state increased, if true - was the delivery provided?,
        # if false - estimate the delivery, if true - was it enough to reach the next day state?
        for point in self.points:
            periods = len(point.demands)
            if periods > 1:
                keys = list(point.demands.keys())
                for i in range(periods - 1):
                    state1, state2 = point.demands[keys[i]][0], point.demands[keys[i + 1]][0]
                    ds = state2 - state1
                    if ds > 0:
                        delivery = point.demands[keys[i + 1]][1]
                        if delivery > 0:
                            if ds < delivery:
                                # TODO: check formula
                                small_delivery = delivery
                                delivery += ds + 0.5 * (point.capacity * point.max_level - state2)
                                point.demands[keys[i]] = (state1, delivery)
                                if verbose:
                                    print("small delivery ({}) {}[{}] = ({},{})".format(
                                        small_delivery, point.ID, keys[i], state1, delivery))
                        else:
                            delivery = ds + 0.5 * (point.capacity * point.max_level - state2)
                            point.demands[keys[i]] = (state1, delivery)
                            if verbose:
                                print("zero delivery {}[{}] = ({},{})".format(
                                    point.ID, keys[i], state1, delivery))

    def read_mtx(self, data_file, excel=True, verbose=False):
        if excel:
            df = pd.read_excel(data_file)
        else:
            df = pd.read_csv(data_file, sep=';', decimal=',')

        for _, row in df.iterrows():
            i, j, d = int(row['id_od']), int(row['Id_do']), float(row['km']) # .replace(',', '.')
            origin, destination = self.get_point(i), self.get_point(j)
            #if origin is not None and destination is not None:
            self.distance_matrix[(i, j)] = d
        # show distances
        if verbose:
            for pair in self.distance_matrix:
                print("d({},{})={}".format(pair[0], pair[1], self.distance_matrix[pair]))

    def load_prepared_data(self, csv_file, verbose=False):
        df = pd.read_csv(csv_file, sep=';')

        self.depot = df['ID'][0]
        self.capacity = -df['Popyt1'][0]
        self.nroutes.append(df['Mozna1'][0])
        self.nroutes.append(df['Mozna2'][0])
        self.nroutes.append(df['Mozna3'][0])

        # initialize sets and usets
        for _ in range(3):
            self.sets.append([])
            self.usets.append([])

        for _, row in df.iterrows():
            point = self.get_point(int(row['ID']))
            if point is not None:
                #print(point.ID)
                point.demands.append(row['Popyt1'])
                point.demands.append(row['Popyt2'])
                point.demands.append(row['Popyt3'])
                if row['Mozna1']:
                    self.sets[0].append(point)
                if row['Mozna2']:
                    self.sets[1].append(point)
                if row['Mozna3']:
                    self.sets[2].append(point)
                if row['Musi1']:
                    self.usets[0].append(point)
                if row['Musi2']:
                    self.usets[1].append(point)
                if row['Musi3']:
                    self.usets[2].append(point)
        if verbose:
            for i in range(len(self.sets)):
                print([p.ID for p in self.sets[i]], [p.ID for p in self.usets[i]])

    def route_distance(self, route):
        dist, N = 0, len(route)
        if N > 0:
            dist = self.distance_matrix[(self.depot, route[0].ID)]
            for i in range(1, N):
                dist += self.distance_matrix[(route[i - 1].ID, route[i].ID)]
            dist += self.distance_matrix[(route[N - 1].ID, self.depot)]
        return dist

    def solution_distance(self, solution):
        distance = 0
        for day in range(len(solution)):
            for route in solution[day]:
                distance += self.route_distance(route)
        return distance

    def generate_initial_solution(self, tries=10):
        days = len(self.nroutes)
        solution = []
        for day in range(days):
            # generate alternative solutions
            day_alternatives = []
            for _ in range(tries):
                day_state = []
                urgents = [p for p in self.usets[day]]
                possibles = [p for p in self.sets[day] if p not in self.usets[day]]
                for _ in range(self.nroutes[day]):
                    day_route = []
                    total = 0
                    # add urgent points
                    while total < self.capacity and len(urgents) > 0:
                        upoint = np.random.choice(urgents, 1)[0]
                        urgents.remove(upoint)
                        day_route.append(upoint)
                        total += upoint.demands[day]
                    # add other possible points
                    while total < self.capacity and len(possibles) > 0:
                        ppoint = np.random.choice(possibles, 1)[0]
                        possibles.remove(ppoint)
                        day_route.append(ppoint)
                        total += ppoint.demands[day]
                    day_state.append(day_route)
                day_alternatives.append(day_state)
            # select best alternative
            best_alternative = None
            best_distance = float('inf')
            for alternative in day_alternatives:
                dist = sum([self.route_distance(route) for route in alternative])
                #print("Alt dist:", dist)
                if dist < best_distance:
                    best_alternative = alternative
                    best_distance = dist
            #print()
            solution.append(best_alternative)
        return solution

    def optimize_solution(self, solution, end_temperature=1e-3):
        opt_solution = []
        for day in range(len(solution)):
            day_routes = []
            for route in solution[day]:
                opt_route = SA_TSP(self, places=route, end_temperature=end_temperature)[-1]
                day_routes.append(opt_route)
            opt_solution.append(day_routes)
        return opt_solution