import networkx as nx
from pyvis.network import Network

def plot_interactive(options, show_options=True):
    # Creating a PG example
    pg = nx.DiGraph()
    pg.add_nodes_from([2, 3])
    pg.add_nodes_from(range(100, 110))
    H = nx.path_graph(10)
    pg.add_nodes_from(H)


    # Pyvis graph
    nt = Network(height=f"400px", width=f"400px", directed=True)

    # Convert networkx graph into pyvis graph
    nt.from_nx(pg)

    # Show options if requested
    if show_options:
        nt.show_buttons()
    else:
        nt.set_options(options)

    # Save the html file
    nt.show('pg.html')

if __name__ == "__main__":

    options = ""
    plot_interactive(show_options=True, options="")
