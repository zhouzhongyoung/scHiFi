"""Differential expression and GO/KEGG analysis of the QX predicted clusters."""
from pathlib import Path
import collections, functools, gzip, hashlib, json, platform
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom, mannwhitneyu, fisher_exact

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'analysis/inputs/groups/QX_Limb_Muscle'
RESULT = HERE / 'statistics'
RESULT.mkdir(exist_ok=True)
DB = HERE / 'databases'

def bh(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted,1)
    return out

def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from schifi.data import load_dataset
    loaded = load_dataset(ROOT / 'data/QX_Limb_Muscle/data.h5')
    saved = pd.read_csv(DATA / 'scHiFi.csv')
    adata = sc.AnnData(loaded.counts)
    adata.var_names = loaded.gene_names.astype(str)
    adata.obs = saved.copy()
    adata.obs_names = saved.cell_id.astype(str).to_numpy()
    assert list(saved.cell_key.astype(str)) == list(adata.obs.cell_key.astype(str))
    predicted = adata.obs.predicted_cluster.astype(str).to_numpy()
    reference = adata.obs.reference_label.astype(str).to_numpy()
    assert np.array_equal(predicted, saved.predicted_cluster.astype(str))
    clusters = sorted(set(predicted), key=int)
    labels = sorted(set(reference), key=int)
    names = dict(zip(reference, adata.obs.reference_type.astype(str)))
    matrix = np.array([[np.sum((predicted==c)&(reference==r)) for r in labels] for c in clusters])
    a,b = linear_sum_assignment(-matrix)
    assignment = {clusters[i]:labels[j] for i,j in zip(a,b)}
    mapping = []
    for i,c in enumerate(clusters):
        r = assignment[c]; j=labels.index(r)
        mapping.append({'predicted_cluster':c,'reference_label':r,'reference_type':names[r],
                        'cluster_cells':int(matrix[i].sum()),'reference_cells':int(matrix[:,j].sum()),
                        'matched_cells':int(matrix[i,j]),'cluster_fraction':float(matrix[i,j]/matrix[i].sum()),
                        'reference_recall':float(matrix[i,j]/matrix[:,j].sum())})
    mapping = pd.DataFrame(mapping)
    assert np.array_equal(np.array([assignment[c] for c in predicted]), saved.hungarian_reference_label.astype(str))
    mapping.to_csv(RESULT/'cluster_to_type.csv',index=False)
    pd.DataFrame(matrix,index=clusters,columns=[names[r] for r in labels]).rename_axis('predicted_cluster').to_csv(RESULT/'confusion_counts.csv')
    obs = pd.DataFrame({'cell_key':adata.obs.cell_key.to_numpy(),'cell_id':adata.obs.cell_id.to_numpy(),
                        'input_row':adata.obs.input_row.to_numpy(),'predicted_cluster':predicted,
                        'reference_type':adata.obs.reference_type.astype(str).to_numpy(),
                        'matched_type':[names[assignment[c]] for c in predicted]})
    obs.to_csv(RESULT/'cell_groups.csv',index=False)
    assert sparse.issparse(adata.X) and adata.X.data.min()>=0
    assert np.all(adata.X.data == np.round(adata.X.data))
    keep = np.asarray((adata.X>0).sum(axis=0)).ravel()>0
    adata=adata[:,keep].copy()
    genes = set(adata.var_names)
    assert len(genes)==16512
    (RESULT/'expressed_gene_universe.txt').write_text('\n'.join(sorted(genes))+'\n')
    sc.pp.normalize_total(adata,target_sum=10000)
    sc.pp.log1p(adata)
    # Remove reference fields from the statistical object to prevent accidental grouping.
    adata.obs=adata.obs[['predicted_cluster']].copy()
    sc.tl.rank_genes_groups(adata,groupby='predicted_cluster',reference='rest',method='wilcoxon',
                           tie_correct=True,corr_method='benjamini-hochberg',use_raw=False)
    deg=sc.get.rank_genes_groups_df(adata,group=None).rename(columns={'group':'predicted_cluster','names':'gene'})
    deg.predicted_cluster=deg.predicted_cluster.astype(str)
    assert len(deg)==len(genes)*6
    deg['significant']=deg.pvals_adj<0.05
    deg['direction']=np.where(deg.significant,np.where(deg.logfoldchanges>0,'up','down'),'ns')
    deg.to_csv(RESULT/'DEG_all_clusters.csv.gz',index=False,compression='gzip')
    summaries=[]
    selected=[]
    input_genes={}
    check=[]
    for c in clusters:
        d=deg[deg.predicted_cluster==c]
        assert np.allclose(bh(d.pvals),d.pvals_adj,atol=1e-12)
        up=d[(d.pvals_adj<0.05)&(d.logfoldchanges>0)]
        down=d[(d.pvals_adj<0.05)&(d.logfoldchanges<0)]
        summaries.append({'predicted_cluster':c,'reference_type':names[assignment[c]],
                          'significant_up':len(up),'significant_down':len(down),'tested_genes':len(d)})
        input_genes[c]=set(up.gene)
        (RESULT/f'C{c}_upregulated_genes.txt').write_text('\n'.join(up.gene)+'\n')
        top=up.sort_values(['scores','gene'],ascending=[False,True]).head(5).copy()
        selected.append(top)
        # Independent statistical spot check: tie-corrected, asymptotic rank test.
        gene=top.iloc[0].gene; idx=adata.var_names.get_loc(gene)
        values=adata.X[:,idx].toarray().ravel()
        p=mannwhitneyu(values[predicted==c], values[predicted!=c],alternative='two-sided',
                       method='asymptotic',use_continuity=False).pvalue
        original=float(top.iloc[0].pvals)
        assert np.isclose(p,original,rtol=1e-4,atol=1e-250),(c,p,original)
        check.append({'cluster':c,'gene':gene,'scanpy_p':original,'scipy_p':float(p)})
    pd.DataFrame(summaries).to_csv(RESULT/'DEG_counts.csv',index=False)
    top=pd.concat(selected,ignore_index=True)
    top.to_csv(RESULT/'heatmap_top5_genes_per_cluster.csv',index=False)
    # Log-expression and per-cell gene-standardized means for the plotting workflow.
    display_genes=list(dict.fromkeys(top.gene.tolist()+['Des','Mfap5','Cd79b','Cdh5','Cd3g','C1qa']))
    display_genes=[g for g in display_genes if g in genes]
    values=adata[:,display_genes].X.toarray()
    sd=values.std(axis=0);sd[sd==0]=1
    z=(values-values.mean(axis=0))/sd
    pd.DataFrame(values,columns=display_genes).to_csv(RESULT/'plot_gene_expression.csv.gz',index=False,compression='gzip')
    pd.DataFrame([z[predicted==c].mean(axis=0) for c in clusters],index=clusters,columns=display_genes).rename_axis('predicted_cluster').to_csv(RESULT/'cluster_mean_gene_z.csv')
    # This embedding visualization is independent of the database downloads.
    if not (RESULT/'umap_coordinates.csv').exists():
        sc.pp.neighbors(adata,n_neighbors=15,use_rep='X_model',metric='euclidean',random_state=0)
        sc.tl.umap(adata,min_dist=0.2,random_state=0)
        pd.DataFrame(adata.obsm['X_umap'],columns=['UMAP1','UMAP2']).to_csv(RESULT/'umap_coordinates.csv',index=False)
    # Ontology parser: positive annotations propagate only over is_a and part_of.
    terms={}; aliases={}; current=None; header=[]
    def flush(t):
        if t and 'id' in t and not t.get('obsolete',False):
            terms[t['id']]=t
            for alt in t['alt']:aliases[alt]=t['id']
    for line in (DB/'go-basic.obo').read_text().splitlines():
        if line=='[Term]':flush(current);current={'parents':[],'alt':[]}
        elif line.startswith('['):flush(current);current=None
        elif current is None:
            if line.startswith(('data-version:','format-version:')):header.append(line)
        elif line.startswith('id: '):current['id']=line[4:]
        elif line.startswith('name: '):current['name']=line[6:]
        elif line.startswith('namespace: '):current['namespace']=line[11:]
        elif line.startswith('alt_id: '):current['alt'].append(line[8:])
        elif line.startswith('is_a: '):current['parents'].append(line[6:].split()[0])
        elif line.startswith('relationship: part_of '):current['parents'].append(line.split()[2])
        elif line=='is_obsolete: true':current['obsolete']=True
    flush(current)
    @functools.lru_cache(None)
    def ancestors(go):
        if go not in terms:return frozenset()
        out={go}
        for parent in terms[go]['parents']:out.update(ancestors(aliases.get(parent,parent)))
        return frozenset(out)
    go_sets=collections.defaultdict(set); gaf_header=[]; evidence=collections.Counter()
    with gzip.open(DB/'MOUSE-mod.gaf.gz','rt') as f:
        for line in f:
            if line.startswith('!'):
                if len(gaf_header)<30:gaf_header.append(line.strip())
                continue
            v=line.rstrip('\n').split('\t')
            if len(v)<15 or 'NOT' in v[3].split('|') or '10090' not in [t.split(':')[-1] for t in v[12].split('|')]:continue
            # Exact mouse symbol matching; no ortholog conversion or ambiguous alias mapping.
            symbol=v[2]
            if symbol not in genes:continue
            evidence[v[6]]+=1
            go=aliases.get(v[4],v[4])
            for term in ancestors(go):go_sets[term].add(symbol)
    libraries={namespace:{go:members for go,members in go_sets.items() if terms[go].get('namespace')==namespace}
               for namespace in ['biological_process','cellular_component','molecular_function']}
    kegg_gene_names={}
    for line in (DB/'kegg_mmu_genes.tsv').read_text().splitlines():
        v=line.split('\t')
        if len(v)>=2:
            # list/mmu has gene ID, location, type and symbol/description.
            description=v[-1]
            symbol=description.split(';')[0].split(',')[0].strip()
            if symbol in genes:kegg_gene_names[v[0]]=symbol
    pathway_names={line.split('\t')[0].replace('path:',''):line.split('\t')[1].replace(' - Mus musculus (house mouse)','')
                   for line in (DB/'kegg_mmu_pathways.tsv').read_text().splitlines() if '\t' in line}
    kegg_sets=collections.defaultdict(set)
    for line in (DB/'kegg_mmu_gene_pathway.tsv').read_text().splitlines():
        gene,pathway=line.split('\t');pathway=pathway.replace('path:','')
        if gene in kegg_gene_names:kegg_sets[pathway].add(kegg_gene_names[gene])
    libraries['KEGG']=dict(kegg_sets)
    enrichment=[]; library_info={}
    for library,all_sets in libraries.items():
        # The expressed-gene universe is intersected with genes annotated in this library.
        universe=set().union(*all_sets.values())
        tested={term:members & universe for term,members in all_sets.items() if 10<=len(members & universe)<=500}
        assert len(tested)>50,(library,len(tested),len(universe))
        library_info[library]={'annotated_expressed_universe':len(universe),'tested_terms':len(tested)}
        (RESULT/f'{library}_universe.txt').write_text('\n'.join(sorted(universe))+'\n')
        (RESULT/f'{library}.gmt').write_text(''.join(term+'\t'+(terms[term]['name'] if library!='KEGG' else pathway_names[term])+'\t'+'\t'.join(sorted(s))+'\n' for term,s in sorted(tested.items())))
        for c in clusters:
            query=input_genes[c]&universe
            rows=[]
            for term,members in tested.items():
                overlap=sorted(query&members)
                p=float(hypergeom.sf(len(overlap)-1,len(universe),len(members),len(query)))
                rows.append({'predicted_cluster':c,'reference_type':names[assignment[c]],'library':library,
                             'term_id':term,'term_name':terms[term]['name'] if library!='KEGG' else pathway_names[term],
                             'universe_size':len(universe),'term_size':len(members),'query_size':len(query),
                             'overlap_count':len(overlap),'gene_ratio':len(overlap)/len(query) if query else 0,
                             'pvalue':p,'overlap_genes':';'.join(overlap)})
            q=bh([r['pvalue'] for r in rows])
            for r,padj in zip(rows,q):r['p_adjust']=float(padj)
            enrichment.extend(rows)
            hit=sorted(rows,key=lambda r:r['pvalue'])[0]
            k=hit['overlap_count'];n=hit['query_size'];m=hit['term_size'];N=hit['universe_size']
            fisher=fisher_exact([[k,n-k],[m-k,N-m-n+k]],alternative='greater').pvalue
            assert np.isclose(fisher,hit['pvalue'],rtol=1e-8,atol=1e-250)
    enrich=pd.DataFrame(enrichment)
    enrich.to_csv(RESULT/'enrichment_all_tests.csv.gz',index=False,compression='gzip')
    enrich[enrich.p_adjust<0.05].sort_values(['library','predicted_cluster','p_adjust','term_id']).to_csv(RESULT/'enrichment_significant.csv',index=False)
    config={'normalization':{'target_sum':10000,'transform':'log1p','minimum_gene_detection_cells':1},
            'DEG':{'groupby':'predicted_cluster','test':'Wilcoxon rank-sum','tie_correct':True,'contrast':'one vs rest',
                   'correction':'BH across 16512 genes separately per cluster','padj_max_exclusive':0.05,
                   'effect_size_threshold':None,'enrichment_input':'padj < 0.05 and estimated log2FC > 0'},
            'enrichment':{'organism':'Mus musculus','test':'one-sided hypergeometric',
                          'universe':'detected genes intersected with the genes annotated in each library',
                          'min_set_size':10,'max_set_size':500,'BH_family':'all eligible terms separately for each cluster and ontology namespace / KEGG',
                          'padj_max_exclusive':0.05,'GO_propagation':['is_a','part_of'],'exclude_NOT':True,'symbol_matching':'exact'},
            'database_headers':{'GO':header,'GAF':gaf_header,'KEGG':(DB/'kegg_pathway_info.txt').read_text()},
            'libraries':library_info,
            'versions':{'python':platform.python_version(),'scanpy':sc.__version__,'numpy':np.__version__,
                        'scipy':scipy.__version__,'pandas':pd.__version__},
            'verification':{'n_cells':len(obs),'n_unique_cell_keys':obs.cell_key.nunique(),'matched_cells':int(matrix[a,b].sum()),
                            'mapping_matches_prepared_data':True,'reference_columns_removed_before_DEG':True,
                            'independent_rank_test_spot_checks':check,'BH_checks_all_clusters':True,'hypergeometric_fisher_checks':24}}
    (HERE/'analysis_settings.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))
    print(mapping.to_string(index=False),flush=True)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(json.dumps(library_info),flush=True)
    print('DONE',flush=True)

if __name__=='__main__':main()
