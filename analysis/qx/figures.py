"""Draw Figures 5 and 6 from the included statistical tables."""
from pathlib import Path
import importlib.util,json,textwrap
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm,LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
PRIOR=HERE/'figure_utils.py'
spec=importlib.util.spec_from_file_location('existing_scientific_inputs',PRIOR)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
AUDIT=HERE/'layout';AUDIT.mkdir(exist_ok=True)
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Liberation Sans','Arial','DejaVu Sans']
plt.rcParams['svg.fonttype']='none'
plt.rcParams.update({'font.size':7,'axes.labelsize':6.5,'axes.linewidth':.65,
                     'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6.5,
                     'pdf.fonttype':42,'legend.frameon':False})
ORDER=p.ORDER
NORM=Normalize(0,25)
SIGNIF=LinearSegmentedColormap.from_list('significance',['#F2DED5','#D78F7B','#B64342','#782A32'])

def panel(fig,x,y,letter):p.text(fig,x,y,f'({letter})',fontsize=8.5,weight='bold')

def cluster_key(fig,y):
    for x,c in zip([11,40,73,103,136,157],ORDER):
        p.strip(fig,x,y-1,2,2,p.COLORS[c])
        p.text(fig,x+2.7,y,f'{c} {p.SHORT[c]}',fontsize=6.4)

def save_pub_set(fig,out_dir,stem,formats=('svg','pdf','png'),dpi=600,bbox_inches=None,close=True):
    """Export vector PDF and SVG, and a 600 dpi PNG."""
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    for ax in fig.axes:
        for item in ax.collections:item.set_rasterized(False)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for item in fig.findobj(mpl.text.Text):
        if not item.get_visible() or not item.get_text().strip():continue
        box=item.get_window_extent(renderer)
        if box.x0<-.7 or box.y0<-.7 or box.x1>fig.bbox.width+.7 or box.y1>fig.bbox.height+.7:
            outside.append(item.get_text())
    assert not outside,outside
    (AUDIT/f'{stem}_layout.json').write_text(json.dumps({'width_mm':178,'height_mm':fig._height_mm,'text_outside_canvas':outside},indent=2))
    for fmt in formats:
        target=out/f'{stem}.{fmt}'
        fig.savefig(target,dpi=dpi,bbox_inches=bbox_inches)
        if fmt=='svg':
            tree=ET.parse(target)
            meshes=[e for e in tree.iter() if e.attrib.get('id')=='cell_heatmap']
            if meshes:
                # Match Agg's non-antialiased cell boundaries, avoiding pale SVG seams.
                for mesh in meshes:mesh.set('shape-rendering','crispEdges')
                ET.register_namespace('','http://www.w3.org/2000/svg')
                ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
                tree.write(target,encoding='utf-8',xml_declaration=True)
    fig.savefig(out/f'{stem}_preview.png',dpi=200)
    if close:plt.close(fig)

def fig5():
    fig=p.canvas(141)
    panel(fig,2,3,'a')
    genes=[g for c in ORDER for g in p.top[p.top.predicted_cluster==c].gene]
    values=p.expr[genes].to_numpy();z=(values-values.mean(0))/values.std(0)
    cell_order=np.concatenate([np.flatnonzero(p.pred==c)[np.lexsort((p.uv[p.pred==c,1],p.uv[p.pred==c,0]))] for c in ORDER])
    ax=p.axes(fig,13,7.5,151,36)
    mesh=ax.pcolormesh(np.arange(31),np.arange(len(cell_order)+1),z[cell_order],cmap=p.HEAT,
                  norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2.5),edgecolors='none',antialiased=False)
    mesh.set_gid('cell_heatmap')
    ax.set_xlim(0,30);ax.set_ylim(len(cell_order),0)
    ax.set_xticks(np.arange(30)+.5,genes,rotation=60,ha='right',fontstyle='italic',fontsize=6.2)
    bounds=np.cumsum([0]+[int((p.pred==c).sum()) for c in ORDER])
    ax.set_yticks((bounds[:-1]+bounds[1:])/2,[str(c) for c in ORDER],fontsize=6.5)
    ax.tick_params(axis='x',length=0,pad=2)
    ax.tick_params(axis='y',length=0,pad=6)
    for sp in ax.spines.values():sp.set_visible(False)
    for j,c in enumerate(ORDER):
        if j>0:ax.axhline(bounds[j],color='white',lw=.7);ax.axvline(5*j,color='white',lw=.7)
        ax.plot([-.0066,-.0066],[bounds[j],bounds[j+1]],transform=ax.get_yaxis_transform(),
                color=p.COLORS[c],lw=2,clip_on=False)
        p.strip(fig,13+151*j/6+.15,5.8,151/6-.3,1.1,p.COLORS[c])
        p.text(fig,13+151*(j+.5)/6,3.3,f'{c} {p.SHORT[c]}',fontsize=6.7,ha='center')
    p.text(fig,3,27,'Cells',rotation=90,ha='center',fontsize=6.5)
    p.colorbar(fig,168,15.5,1.5,18,p.HEAT,TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2.5),[-2,0,2.5],'z score')
    panel(fig,2,58,'b')
    handles=[Line2D([],[],marker='o',ls='none',ms=2.6,color=color,label=label)
             for label,color in [('Up',p.RED),('Down',p.BLUE),('Other','#C5C9CD')]]
    fig.legend(handles=handles,loc='center right',bbox_to_anchor=(.97,1-58/141),
               ncol=3,columnspacing=.8,handletextpad=.25,handlelength=.65,fontsize=6.5,borderaxespad=0)
    for j,c in enumerate(ORDER):
        x0=12+j*27.4;ax=p.axes(fig,x0,69,23,26.5)
        d=p.deg[p.deg.predicted_cluster==c]
        x=d.logfoldchanges.to_numpy();y=-np.log10(np.maximum(d.pvals_adj.to_numpy(),1e-300))
        for direction,color,alpha,size in [('ns','#C5C9CD',.55,.5),('down',p.BLUE,.38,.65),('up',p.RED,.5,.7)]:
            sel=d.direction.to_numpy()==direction
            ax.scatter(np.clip(x[sel],-12,12),y[sel],s=size,color=color,alpha=alpha,linewidths=0)
        ax.axvline(0,color='#B7BEC4',lw=.4,ls=(0,(2,2)))
        ax.axhline(-np.log10(.05),color='#999999',lw=.4,ls=(0,(2,2)))
        ax.set_xlim(-12.7,12.7);ax.set_ylim(0,315);ax.set_xticks([-10,0,10]);ax.set_yticks([0,100,200,300])
        if j>0:ax.tick_params(labelleft=False,left=False)
        ax.grid(axis='y',color=p.GRID,lw=.35);p.tidy(ax)
        lead=p.top[p.top.predicted_cluster==c].iloc[0]
        ax.annotate(lead.gene,(np.clip(lead.logfoldchanges,-12,12),-np.log10(max(float(lead.pvals_adj),1e-300))),
                    xytext=(11,270),ha='right',va='center',fontsize=6.2,fontstyle='italic',
                    arrowprops={'arrowstyle':'-','lw':.35,'color':'#777777'},
                    bbox={'facecolor':'white','edgecolor':'none','pad':.2,'alpha':.9})
        p.text(fig,x0+11.5,62.9,f'{c} {p.SHORT[c]}',ha='center',fontsize=6.7,weight='bold')
        p.text(fig,x0+11.5,66.4,f'{p.counts.loc[c,"significant_up"]:,} up / {p.counts.loc[c,"significant_down"]:,} down',ha='center',fontsize=6.1,color=p.MUTED)
    p.text(fig,2,82,r'$-\log_{10}(q)$',rotation=90,ha='center',fontsize=6.7)
    p.text(fig,95,100.5,r'Estimated $\log_2$ fold change',ha='center',fontsize=6.8)
    panel(fig,2,105,'c')
    markers=[None,'Cdh5','Mfap5','Des','C1qa','Cd3g','Cd79b']
    for j,gene in enumerate(markers):
        ax=p.axes(fig,11+j*23.4,110.5,21.1,22.5)
        if gene is None:
            for c in ORDER:
                mask=p.pred==c;ax.scatter(p.uv[mask,0],p.uv[mask,1],s=.55,c=p.COLORS[c],linewidths=0)
                ax.text(*np.median(p.uv[mask],axis=0),str(c),ha='center',va='center',fontsize=6.3,weight='bold',
                        bbox={'facecolor':'white','edgecolor':'none','pad':.35,'alpha':.9})
            title='Clusters'
        else:
            v=p.expr[gene].to_numpy();ix=np.argsort(v,kind='stable')
            ax.scatter(p.uv[ix,0],p.uv[ix,1],c=v[ix],cmap=p.EXPR,norm=Normalize(0,4),s=.6,linewidths=0)
            title=gene
        p.text(fig,21.5+j*23.4,106.5,title,ha='center',fontsize=7,fontstyle='italic' if gene else 'normal')
        ax.set_xlim(p.uv[:,0].min()-1,p.uv[:,0].max()+1);ax.set_ylim(p.uv[:,1].min()-1,p.uv[:,1].max()+1)
        ax.set_aspect('equal');ax.axis('off')
    p.colorbar(fig,173.8,113,1.3,16,p.EXPR,Normalize(0,4),[0,4],'')
    cluster_key(fig,137)
    pd.DataFrame({'display_row':range(len(cell_order)),'original_row':cell_order,
                  'cell_key':p.groups.cell_key.iloc[cell_order].to_numpy(),'predicted_cluster':p.pred[cell_order]}).to_csv(AUDIT/'Fig5_heatmap_cell_order.csv',index=False)
    pd.DataFrame({'gene':genes,'module_cluster':np.repeat(ORDER,5)}).to_csv(AUDIT/'Fig5_heatmap_genes.csv',index=False)
    save_pub_set(fig,OUT,'Fig5_QX_compact')

def get_test(library,c,name):
    r=p.enrich[(p.enrich.library==library)&(p.enrich.predicted_cluster==c)&(p.enrich.term_name==name)]
    assert len(r)==1 and r.iloc[0].p_adjust<.05,(library,c,name)
    return r.iloc[0].copy()

def fig6():
    fig=p.canvas(158);panel(fig,2,3,'a');panel(fig,100,3,'b')
    rank_tables=[]
    for k,c in enumerate(ORDER):
        x=12+(k%3)*29;y=11+(k//3)*32
        ax=p.axes(fig,x,y,24,23)
        d=p.deg[(p.deg.predicted_cluster==c)&(p.deg.pvals_adj<.05)&(p.deg.logfoldchanges>0)].sort_values(['scores','gene'],ascending=[False,True]).head(8).copy()
        d['display_rank']=np.arange(1,9);rank_tables.append(d)
        scores=d.scores.to_numpy();span=max(float(np.ptp(scores)),1)
        ax.plot(np.arange(1,9),scores,color='#9BA7AF',lw=.4,zorder=1)
        ax.scatter(np.arange(1,9),scores,s=5,color=p.BLUE,linewidths=0,zorder=2)
        for j,r in enumerate(d.itertuples()):
            ax.annotate(r.gene,(j+1,r.scores),xytext=(1,2),textcoords='offset points',
                        rotation=90,ha='left',va='bottom',fontsize=5.7,fontstyle='italic')
        ax.set_xlim(.4,8.9);ax.set_ylim(scores.min()-.15*span,scores.max()+1.20*span)
        ax.set_xticks([1,4,8]);ax.tick_params(labelsize=5.7,length=1.5,pad=1.5)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3));p.tidy(ax)
        if k%3==0:ax.set_ylabel('Score',fontsize=6,labelpad=1)
        if k//3==1:ax.set_xlabel('Rank',fontsize=6,labelpad=1)
        p.text(fig,x+12,y-2.8,f'{c} {p.SHORT[c]}',ha='center',fontsize=6.4)
    bp=[(0,'blood vessel morphogenesis'),(1,'extracellular matrix organization'),
        (2,'skeletal muscle tissue development'),(2,'ribonucleoprotein complex biogenesis'),
        (5,'immune effector process'),(5,'endocytosis'),(3,'T cell activation'),(4,'B cell activation')]
    bar=[]
    for c,name in bp:bar.append(get_test('biological_process',c,name))
    for c,name in [(1,'extracellular matrix'),(5,'external side of plasma membrane')]:bar.append(get_test('cellular_component',c,name))
    for c,name in [(1,'extracellular matrix structural constituent'),(5,'antigen binding')]:bar.append(get_test('molecular_function',c,name))
    ax=p.axes(fig,145,9,28,58);n=len(bar)
    for j,r in enumerate(bar):
        ax.barh(j,r.overlap_count,height=.74,color=SIGNIF(NORM(-np.log10(r.p_adjust))),edgecolor='none')
        label=r.term_name+f' ({int(r.predicted_cluster)})'
        p.text(fig,143,9+58*(j+.5)/n,'\n'.join(textwrap.wrap(label,width=32,break_long_words=False)),
               fontsize=6.0,ha='right',linespacing=1.0)
    ax.set_xlim(0,230);ax.set_ylim(n-.5,-.5);ax.set_yticks([]);ax.set_xticks([0,100,200])
    ax.set_xlabel('Overlap genes',fontsize=6.4,labelpad=1);p.tidy(ax)
    for lo,hi,label in [(0,8,'BP'),(8,10,'CC'),(10,12,'MF')]:
        p.strip(fig,174.1,9+58*lo/n+.2,1.6,58*(hi-lo)/n-.4,'#E5E5E5')
        p.text(fig,174.9,9+58*(lo+hi)/(2*n),label,rotation=270,ha='center',fontsize=5.6)
    # Shared count and significance keys occupy the gutter, rather than data areas.
    handles=[plt.scatter([],[],s=.4*k,facecolors='#A6A6A6',edgecolors='#666666',lw=.3) for k in [25,75,150]]
    p.text(fig,47,85,'Overlap genes',ha='right',fontsize=6.0)
    fig.legend(handles,[str(k) for k in [25,75,150]],loc='center left',
               bbox_to_anchor=(.28,1-85/158),ncol=3,columnspacing=.8,handletextpad=.3,
               borderaxespad=0,fontsize=6.0)
    cb=p.colorbar(fig,141,73.5,30,1.7,SIGNIF,NORM,[0,12.5,25],r'$-\log_{10}(q)$',orientation='horizontal')
    cb.ax.tick_params(labelsize=5.8,pad=1);cb.ax.set_xticklabels(['0','12.5','≥25']);cb.ax.xaxis.label.set_size(6.2)
    panel(fig,2,84,'c');panel(fig,100,84,'d')
    # GO cluster-specific patterns: all significant associations of selected terms.
    terms=[]
    for c in ORDER:
        for name in p.PROGRAMS['biological_process'][c]:terms.append(get_test('biological_process',c,name))
    ax=p.axes(fig,51,91,43,53);shown=[]
    for j,r in enumerate(terms):
        p.text(fig,49,91+53*(j+.5)/len(terms),'\n'.join(textwrap.wrap(r.term_name,width=30,break_long_words=False)),ha='right',fontsize=6.0,linespacing=1.0)
        for k,c in enumerate(ORDER):
            match=p.enrich[(p.enrich.library=='biological_process')&(p.enrich.term_id==r.term_id)&(p.enrich.predicted_cluster==c)].iloc[0]
            if match.p_adjust>=.05:continue
            ax.scatter(k,j,s=.4*match.overlap_count,c=[-np.log10(match.p_adjust)],cmap=SIGNIF,norm=NORM,
                       edgecolors='#6E4F4C',linewidths=.25,zorder=3)
            shown.append(match)
    ax.set_xlim(-.6,5.6);ax.set_ylim(len(terms)-.5,-.5);ax.set_yticks([])
    ax.set_xticks(range(6),[str(c) for c in ORDER]);ax.set_xlabel('Predicted cluster',fontsize=6.4,labelpad=1)
    ax.grid(axis='both',color='#E2E2E2',lw=.4,ls=(0,(2,2)));ax.set_axisbelow(True);p.tidy(ax)
    # KEGG enrichment ranked by significance, with explicit source cluster IDs.
    paths=[]
    for c in ORDER:
        for name in p.PROGRAMS['KEGG'][c]:
            if name in ['Hedgehog signaling pathway','Spliceosome']:continue
            paths.append(get_test('KEGG',c,name))
    paths=pd.DataFrame(paths).sort_values(['p_adjust','term_id'])
    ax=p.axes(fig,149,91,24,53)
    for j,r in enumerate(paths.itertuples()):
        score=-np.log10(r.p_adjust)
        ax.scatter(score,j,s=.4*r.overlap_count,c=[score],cmap=SIGNIF,norm=NORM,edgecolors='#6E4F4C',lw=.25,zorder=3)
        label=r.term_name+f' ({r.predicted_cluster})'
        p.text(fig,147,91+53*(j+.5)/len(paths),'\n'.join(textwrap.wrap(label,width=32,break_long_words=False)),ha='right',fontsize=6.0,linespacing=1.0)
    ax.set_xlim(2,42);ax.set_ylim(len(paths)-.5,-.5);ax.set_yticks([]);ax.set_xticks([5,20,35])
    ax.set_xlabel(r'$-\log_{10}(q)$',fontsize=6.4,labelpad=1)
    ax.grid(axis='x',color='#E2E2E2',lw=.4,ls=(0,(2,2)));ax.set_axisbelow(True);p.tidy(ax)
    # Type names are a lookup key, not a categorical color encoding in this figure.
    footer=[p.text(fig,0,154,f'{c} {p.SHORT[c]}',fontsize=6.4,color=p.INK) for c in ORDER]
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    widths=[item.get_window_extent(renderer).width/fig.bbox.width*178 for item in footer]
    gap=4.0
    total=sum(widths)+gap*(len(footer)-1)
    x=(178-total)/2
    for item,width in zip(footer,widths):
        item.set_position((x/178,1-154/158))
        x+=width+gap
    (AUDIT/'Fig6_footer_layout.json').write_text(json.dumps({'center_mm':89,'total_width_mm':total,'gap_mm':gap},indent=2))
    pd.concat(rank_tables).to_csv(AUDIT/'Fig6a_ranked_genes.csv',index=False)
    pd.DataFrame(bar).to_csv(AUDIT/'Fig6b_GO_category_bars.csv',index=False)
    pd.DataFrame(shown).to_csv(AUDIT/'Fig6c_GO_cluster_associations.csv',index=False)
    paths.to_csv(AUDIT/'Fig6d_KEGG_ranked_terms.csv',index=False)
    save_pub_set(fig,OUT,'Fig6_QX_four_panel')

if __name__=='__main__':
    import sys
    if len(sys.argv)==1 or '5' in sys.argv[1:]:fig5()
    if len(sys.argv)==1 or '6' in sys.argv[1:]:fig6()
