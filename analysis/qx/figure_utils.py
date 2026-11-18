"""Figure drawing helpers and QX statistical inputs."""
from pathlib import Path
import json, hashlib, textwrap, gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
DATA=HERE/'statistics'
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
AUDIT=HERE/'layout';AUDIT.mkdir(exist_ok=True)
MM=1/25.4
FONT='Liberation Sans'
plt.rcParams.update({'font.family':FONT,'font.size':7.5,'axes.labelsize':7.5,
                     'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                     'axes.linewidth':.55,'xtick.major.width':.5,'ytick.major.width':.5,
                     'xtick.major.size':2,'ytick.major.size':2,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'savefig.facecolor':'white','figure.facecolor':'white',
                     'mathtext.fontset':'custom','mathtext.rm':FONT,
                     'mathtext.it':FONT+':italic','mathtext.bf':FONT+':bold',
                     'mathtext.cal':FONT})
ORDER=[0,1,2,5,3,4]
SHORT={0:'Endothelial',1:'Mesenchymal',2:'Satellite',5:'Macrophage',3:'T cell',4:'B cell'}
ABBR={0:'Endo.',1:'MSC',2:'Satellite',5:'Macro.',3:'T cell',4:'B cell'}
COLORS={0:'#4E6577',1:'#A0927D',2:'#8C7D87',5:'#9E6B63',3:'#7E8B77',4:'#9F945B'}
INK='#242A30';MUTED='#6A727B';GRID='#E8EBED'
BLUE='#527D9B';RED='#B95051'
HEAT=LinearSegmentedColormap.from_list('quiet_expression',[(0,'#416787'),(.30,'#B3C8D5'),(.5,'#FCFBF8'),(.72,'#D7A5A0'),(1,'#9B3B43')])
ENRICH=LinearSegmentedColormap.from_list('enrichment',['#E9EFF1','#9CB3BF','#52778C','#1D4056'])
EXPR=LinearSegmentedColormap.from_list('expression',['#ECECED','#DAC1BF','#B47A77','#7A333B'])
groups=pd.read_csv(DATA/'cell_groups.csv')
pred=groups.predicted_cluster.to_numpy()
deg=pd.read_csv(DATA/'DEG_all_clusters.csv.gz')
top=pd.read_csv(DATA/'heatmap_top5_genes_per_cluster.csv')
counts=pd.read_csv(DATA/'DEG_counts.csv').set_index('predicted_cluster')
expr=pd.read_csv(DATA/'plot_gene_expression.csv.gz')
uv=pd.read_csv(DATA/'umap_coordinates.csv').to_numpy()
enrich=pd.read_csv(DATA/'enrichment_all_tests.csv.gz')
objects=[]

def canvas(height):
    fig=plt.figure(figsize=(178*MM,height*MM))
    fig._height_mm=height
    return fig

def axes(fig,x,y,w,h):
    return fig.add_axes([x/178,1-(y+h)/fig._height_mm,w/178,h/fig._height_mm])

def text(fig,x,y,s,**kwargs):
    defaults={'fontsize':7.5,'color':INK,'ha':'left','va':'center'};defaults.update(kwargs)
    obj=fig.text(x/178,1-y/fig._height_mm,s,**defaults);objects.append(obj)
    return obj

def panel(fig,x,y,letter,title):
    text(fig,x,y,f'({letter})',fontsize=10.5,weight='bold')

def colorbar(fig,x,y,w,h,cmap,norm,ticks,label,orientation='vertical'):
    ax=axes(fig,x,y,w,h)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ax,orientation=orientation)
    cb.outline.set_visible(False)
    cb.set_ticks(ticks);cb.ax.tick_params(labelsize=6.7,width=.4,length=1.6,pad=2)
    if orientation=='horizontal':
        cb.set_label(label,fontsize=7,labelpad=2)
    else:
        ax.set_title(label,fontsize=7,pad=5,loc='center')
    if cb.solids is not None:
        cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.solids.set_antialiased(False)
    return cb

def strip(fig,x,y,w,h,color):
    ax=axes(fig,x,y,w,h);ax.add_patch(Rectangle((0,0),1,1,facecolor=color,edgecolor='none'));ax.axis('off')

def tidy(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#626B73')
    ax.tick_params(colors='#4F5860',pad=2)

def cluster_key(fig,y):
    for x,c in zip([12,41,74,104,137,158],ORDER):
        strip(fig,x,y-1,2.5,2.5,COLORS[c])
        text(fig,x+3.5,y+.2,SHORT[c],fontsize=7)

def save(fig,name):
    # Keep all heatmap cells and scatter points genuinely editable in the SVG.
    for ax in fig.axes:
        for obj in ax.collections:obj.set_rasterized(False)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text().strip():continue
        box=item.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:
            outside.append({'text':item.get_text(),'bbox':list(box.bounds)})
    (AUDIT/f'{name}_layout.json').write_text(json.dumps({'width_mm':178,'height_mm':fig._height_mm,
                                                       'text_outside_canvas':outside},ensure_ascii=False,indent=2))
    assert not outside,outside
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png',dpi=600)
    fig.savefig(OUT/f'{name}_preview.png',dpi=200)
    plt.close(fig)

PROGRAMS={
    'biological_process':{
        0:['blood vessel morphogenesis','endothelium development'],
        1:['extracellular matrix organization','collagen fibril organization'],
        2:['skeletal muscle tissue development','ribonucleoprotein complex biogenesis'],
        5:['immune effector process','endocytosis'],
        3:['T cell activation','T cell receptor signaling pathway'],
        4:['B cell activation','B cell receptor signaling pathway']},
    'KEGG':{
        0:['Leukocyte transendothelial migration','Rap1 signaling pathway'],
        1:['Protein processing in endoplasmic reticulum','Hedgehog signaling pathway'],
        2:['Ribosome','Spliceosome'],
        5:['Phagocytosis','Lysosome biogenesis'],
        3:['T cell receptor signaling pathway','Th1 and Th2 cell differentiation'],
        4:['B cell receptor signaling pathway','Intestinal immune network for IgA production']}}
