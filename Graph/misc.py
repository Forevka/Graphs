from matplotlib.patches import FancyArrowPatch, Circle
import networkx as nx
from . import register_plugin


@register_plugin
def get_literal(num):
    return chr(64+num)

@register_plugin
def getKeyID(custom):
    return custom.id

@register_plugin
def getKeyWeight(custom):
    return custom.weight

def get_link(links, from_id, to_id):
    for link in links:
        if link.from_id == from_id and link.to_id == to_id:
            return link

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)
    
@register_plugin
def draw_network(G, pos, ax, links, sg=None, path_links = None):
    for n in G:
        c=Circle(pos[n],radius=0.05,alpha=0.5)
        ax.add_patch(c)
        G.node[n]['patch']=c
        x,y=pos[n]
    seen={}

    for (u,v,d) in G.edges(data=True):
        n1=G.node[u]['patch']
        n2=G.node[v]['patch']


        rad=0.1
        if (u,v) in seen:
            rad=seen.get((u,v))
            rad=(rad+np.sign(rad)*0.1)*-1
        alpha=0.5
        color = 'k'
        if path_links:
            link = get_link(links, u, v)
            if link in path_links:
                color='r'
        e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                            label = '123',
                            arrowstyle='-|>',
                            connectionstyle='arc3,rad=%s'%rad,
                            mutation_scale=20.0,
                            lw=2,
                            alpha=alpha,
                            color=color)
        #centroid = ((n1.get_center()[0] + n2.get_center()[0]) / 2, (n1.get_center()[1]+n2.get_center()[1]) / 2)

        ax.add_patch(e)
    return e
