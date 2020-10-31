from mllib.trees.decision_trees import DecisionTreeID3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def generic_test_ID3():
    dt = DecisionTreeID3()
    df = pd.DataFrame({'F1': ['a1', 'a2', 'a1', 'a3', 'a2'],
                       'F2': ['b2', 'b1', 'b1', 'b2', 'b2'],
                       'F3': ['c3', 'c3', 'c1', 'c2', 'c2']})
    dt.fit(inputs=df,
           targets=['+', '+', '-', '+', '-'])

    dt.predict()


def play_tennis_test():
    outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(
        ',')
    temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
    humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
    windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
    play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')

    dataset = {'outlook': outlook, 'temp': temp, 'humidity': humidity, 'windy': windy}
    df = pd.DataFrame(dataset, columns=['outlook', 'temp', 'humidity', 'windy'])

    dt = DecisionTreeID3()
    dt.fit(inputs=df,
           targets=play)

    graph, prediction = dt.predict({'outlook': 'overcast', 'temp': 'cool', 'humidity': 'normal', 'windy': 'TRUE'})

    print(prediction)
    # print(find_path(graph, 'outlook', 'humidity', []))

    # G = nx.DiGraph(graph)
    # nx.draw(G)
    # plt.show()


# def find_path(graph, start, end, soFar):
#     soFar = soFar + [start]
#     if start == end:
#         return soFar
#     elif start not in graph:
#         return None
#     for node in graph[start]:
#         if node not in soFar:
#             newPath = find_path(graph[start], node, end, soFar)
#             return newPath
#     return None

def test_party():
    deadline = 'urgent,urgent,near,none,none,none,near,near,near,urgent'.split(',')
    party = 'yes,no,yes,yes,no,yes,no,no,yes,no'.split(',')
    lazy = 'yes,yes,yes,no,yes,no,no,yes,yes,no'.split(',')
    activity = 'party,study,party,party,pub,party,study,tv,party,study'.split(',')

    dataset = {'deadline': deadline, 'party': party, 'lazy': lazy}
    df = pd.DataFrame(dataset, columns=['deadline', 'party', 'lazy'])

    dt = DecisionTreeID3()
    dt.fit(inputs=df,
           targets=activity)

    graph, prediction = dt.predict({'deadline': 'near', 'party': 'yes', 'lazy': 'no'})
    print(prediction)

# generic_test_ID3()
# play_tennis_test()
test_party()
