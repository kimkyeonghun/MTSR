import os
import json
import numpy as np

def build_graph_from_wiki(market_name, connection_f, tic_wiki_f, sel_wiki_f):
    """
        market_name: stock exchange(NASDAQ etc..)
        connection_f: defined connection between companies
        tic_wiki_f: (stock, tickers)
        sel_wiki_f: selected connection from wiki
    """
    tickers = np.genfromtxt(tic_wiki_f, dtype = str, delimiter=',', skip_header=False)
    print("#Selected Tickers:", tickers.shape)

    wiki_ticker_dic = {}
    for idx, t in enumerate(tickers):
        if not t[-1] == 'unknown':
            wiki_ticker_dic[t[-1]] = idx

    print("#Aligned Tikcers:", len(wiki_ticker_dic))

    sel_wiki = np.genfromtxt(sel_wiki_f, dtype = str, delimiter=' ', skip_header=False)
    print("#Selected paths:", len(sel_wiki))
    sel_wiki = set(sel_wiki[:, 0])

    with open(connection_f, 'r') as in_f:
        connections = json.load(in_f)
    print("#Connection Items:", len(connections))

    pruned_paths = set()
    for _, con in connections.items():
        for _, paths in con.items():
            for p in paths:
                key = '_'.join(p)
                if key in sel_wiki:
                    pruned_paths.add(key)

    valid_path = {}
    for idx, path in enumerate(pruned_paths):
        valid_path[path] = idx
        print(path, idx)
    print("#Valid Paths:", len(valid_path))

    wiki_relation_embedding = np.zeros(
        [tickers.shape[0], tickers.shape[0], len(valid_path) + 1],
        dtype=int
    )

    connection_count = 0
    for start, con in connections.items():
        for end, paths in con.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in valid_path.keys():
                    source = wiki_ticker_dic[start]
                    target = wiki_ticker_dic[end]
                    edge = valid_path[path_key]
                    wiki_relation_embedding[source][target][edge] = 1
                    connection_count +=1
    print('connections count:', connection_count,
     'ratio:', connection_count / float(tickers.shape[0] * tickers.shape[0]))

    for i in range(tickers.shape[0]):
        wiki_relation_embedding[i][i][-1] = 1
    
    print(wiki_relation_embedding.shape)
    np.save(market_name + "_wiki_relation", wiki_relation_embedding)
    print("Save completed")

if __name__ == "__main__":
    data_path = "../data/wikidata"
    market_name = 'NASDAQ'
    build_graph_from_wiki(market_name,
                        os.path.join(data_path, market_name + '_connections.json'),
                        os.path.join(data_path, market_name + '_wiki.csv'),
                        os.path.join(data_path, 'selected_wiki_connections.csv'))