"""
Compute n by n similarity of all representations, and save to disk

Uses Facebook's fast query FAISS: https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
"""

import numpy as np
import faiss                   

import h5py
import glob

from pathlib import Path
import time
import math
import os
import sys
import argparse

from ssl_legacysurvey.utils import load_data

def parse_arguments():
    """
    Parse commandline arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')

    parser.add_argument("--data_path", type=str, default='/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/south/',
                        help="Path to directory containing representations")

    parser.add_argument("--rep_dir", type=str, default='../trained_models/test/representations/compiled/',
                        help="Path to directory containing representations")

    parser.add_argument("--output_dir", type=str, default="../trained_models/test/representations/similarity/",
                        help="Subdirectory to save similarity arrays in")
    
    parser.add_argument("--overwrite",action="store_true",
                        help="Overwrite similarity arrays located at <output_dir>")
    
    parser.add_argument("--knearest", type=int, default=1000,
                        help="Number of nearest representations to search for")

    parser.add_argument("--delta_mag", type=float, default=0.5,
                        help="Only search within certain magnitude range of queries to speed up nxn")

    parser.add_argument("--start_on_chunk", type=int, default=0,
                        help="Chunk to start on")

    parser.add_argument("--survey", type=str, default="south",
                        help="DESI legacy imaging survey subset")

    parser.add_argument("--rep_file_head", type=str, default="representations",
                        help="File head of saved chunks")

    parser.add_argument("--decals_dir", type=str, default="/global/cfs/projectdirs/cusp/LBL/decals_galaxy_survey/",
                        help="Directory of DESI catalogues/images")
    
    parser.add_argument("--ngals_tot", type=int, default=62000000, #42272646,
                        help="Total number of data files, if data_path is directory, not h5")
    
    parser.add_argument("--chunksize_similarity", type=int, default=1000000,
                        help="Output chunksize")

    parser.add_argument("--sim_chunksize", type=int, default=1000000,
                        help="Size of chunks for similarity computation. Requires smaller if low-memory machine")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    return args

def get_dataset_size(params):
    """Find total size of data.
    Determined by the total number of samples in all .npy files in specified representation directory"""
    all_rep_files = sorted(
        glob.glob(
            os.path.join(
                params['rep_dir'],
                params['rep_file_head'],
                ) + "*00.npy")
        )
    
    ngals_tot = 0
    for i, f in enumerate(all_rep_files):
        d = np.load(f, mmap_mode='r')
        ngals_tot += d.shape[0]
        
        if i==0:
            chunksize=ngals_tot
            rep_dim = d.shape[1]
            
    return chunksize, rep_dim, ngals_tot

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def load_rep_chunk(rep_file):
    """
    Load chunk of representations

    Normalize along feature dimension if desired
    """
    rep_file_norm = os.path.splitext(rep_file)[0]+'_normalized.npy' 

    # if the normalized version exists then load that
    if os.path.isfile(rep_file_norm):
        rep = np.load(rep_file_norm, mmap_mode='r')
        
    # else make it and save to disk
    else:
        rep = np.load(rep_file)    
        rep = rep/np.linalg.norm(rep, axis=1, keepdims=True)
        np.save(rep_file_norm, rep)

    return rep

def load_representatons_by_inds(params, inds_query):
    nquery = inds_query.shape[0]
    rep_query = np.zeros((nquery, params['rep_dim']), dtype=np.float32)
    nquery_start = 0
    
    for ichunk in range(params['nchunks']):
        if params['verbose']:
            print('Getting representations from chunk: ', ichunk)
        
        igal_start = ichunk*params['chunksize']
        igal_end = min(igal_start+params['chunksize'], params['ngals_tot'])
        
        if igal_start > inds_query[-1] or igal_end < inds_query[0]:
            # since inds_query is monotonically increasing there is no overlap
            continue

        # load and save normalization
        rep_file = f"{params['rep_dir']}{params['rep_file_head']}_{igal_start:09d}_{igal_end:09d}.npy"
    
        rep = load_rep_chunk(rep_file)
        
        dm_query = (inds_query >= igal_start) & (inds_query < igal_end)
        inds_query_chunk = inds_query[dm_query] % params['chunksize']
        
        nquery_end = nquery_start + inds_query_chunk.shape[0]
        print('number of queries in chunk {:d}:'.format(ichunk), nquery_end - nquery_start)

        rep_query[nquery_start:nquery_end] = rep[inds_query_chunk]
        
        nquery_start = nquery_end

    return rep_query

def calculate_similarity_for_chunk(params, rep_query, chunk_min, chunk_max):
    """
    Loop through all other chunks and compute similarity to query representations
    """
    nquery = rep_query.shape[0]
    
    distance = np.zeros((nquery, params['knearest']), dtype=np.float32)
    similarity_indexes = np.zeros((nquery, params['knearest']), dtype=np.int32)

    if params['use_gpu']:
        ngpus = faiss.get_num_gpus()
        print('Number of GPUs = ', ngpus)

        if ngpus == 1:
            res = faiss.StandardGpuResources()

    for ichunk in range(chunk_min, chunk_max):

        print('Finding similar galaxies in chunk: ', ichunk)
    
        igal_start = ichunk*params['chunksize']
        igal_end = min(igal_start+params['chunksize'], params['ngals_tot'])
        
        rep_file = f"{params['rep_dir']}{params['rep_file_head']}_{igal_start:09d}_{igal_end:09d}.npy"

        rep = load_rep_chunk(rep_file)
        rep = rep[:] # load in if mmap_mode='r'
        print('loaded chunk', rep.shape, rep_query.shape)
        d = rep.shape[-1]
    
        # full million can crash faiss index. so again do in smaller chunks
        sim_nchunks = math.ceil(rep.shape[0]/params['sim_chunksize'])
        for i in range(sim_nchunks):
            tstart = time.time()

            ichunk_start = i*params['sim_chunksize']
            ichunk_end = ichunk_start + params['sim_chunksize']
            
            # similarity search through chunk
            if params['use_faiss']:
                index = faiss.IndexFlatIP(d) #  distance
                if params['use_gpu']:
                    if ngpus==1:
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                    else:
                        index = faiss.index_cpu_to_all_gpus(index)
                        
                index.add(rep[ichunk_start:ichunk_end]) # add vectors to the index
                print('total number of entries in database =', index.ntotal)
            
                # search database for closest indexes to queries  
                dist_chunk, inds_chunk = index.search(rep_query, params['knearest']) # sanity check

            else:
                dist_chunk = np.inner(rep_query, rep[ichunk_start:ichunk_end])
                inds_chunk = np.argsort(dist_chunk, axis=1)[:, ::-1]
                dist_chunk = np.take_along_axis(dist_chunk, inds_chunk, axis=1)
                
            inds_chunk += igal_start + ichunk_start

            # add to already assembeled list, keeping only knearest top similarity
            dist_all = np.concatenate((distance, dist_chunk), axis=1)
            inds_all = np.concatenate((similarity_indexes, inds_chunk), axis=1)
    
            ind_sort = np.argsort(dist_all, axis=1)[:, ::-1]

            distance = np.take_along_axis(dist_all, ind_sort, axis=1)[:, :params['knearest']]
            similarity_indexes = np.take_along_axis(inds_all, ind_sort, axis=1)[:, :params['knearest']]

            print(distance, similarity_indexes)
            #print(distance, similarity_indexes)
            print('time elapsed', time.time()-tstart)

    return similarity_indexes, distance

def main(args):

    params = vars(args)
    if params['verbose']:
        print(params)

    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)

    params['use_faiss'] = True
    params['use_gpu'] = True
    params['norm'] = True # normalize vector lengths

    if not params['use_faiss']:
        params['use_gpu'] = False

    params['chunksize'], params['rep_dim'], ngals_tot = get_dataset_size(params)
    params['ngals_tot'] = min(params['ngals_tot'], ngals_tot)
    
    print(f"\nNumber of galaxies={params['ngals_tot']}, chunksize={params['chunksize']}, representation_dim={params['rep_dim']}\n")

    # load in representations in chunks
    params['nchunks'] = int(math.ceil(params['ngals_tot']/params['chunksize']))
    params['nchunks_similarity'] = int(math.ceil(params['ngals_tot']/params['chunksize_similarity'])) + 1

    # Search within certain magnitude range of queries requires magnitude
    DDL = load_data.DecalsDataLoader(image_dir=params['data_path'])

    print("\nloading flux of all galaxies in survey. Be patient.\n")

    gals = DDL.get_data(-1, fields=['flux'])
    zmag = gals['mag_z']

    for ichunk_similarity in range(params['start_on_chunk'], params['nchunks_similarity']):

        print('Running on query chunk: ', ichunk_similarity)
    
        rep_query_start = ichunk_similarity*params['chunksize_similarity']
        rep_query_end = min(rep_query_start+params['chunksize_similarity'], params['ngals_tot'])

        inds_file_out = os.path.join(
            params['output_dir'],
            f"inds_knearest{params['knearest']:03d}_{rep_query_start:09d}_{rep_query_end:09d}.npy",
            )
        
        dist_file_out = os.path.join(
            params['output_dir'],
            f"dist_knearest{params['knearest']:03d}_{rep_query_start:09d}_{rep_query_end:09d}.npy",
            )

        file_exists = os.path.isfile(inds_file_out) and os.path.isfile(dist_file_out)
        if not params['overwrite'] and file_exists:
            continue
        
        np.save(inds_file_out, np.array([0])) # create output file first
        np.save(dist_file_out, np.array([0]))
    
        inds_query = np.arange(rep_query_start, rep_query_end)

        rep_query = load_representatons_by_inds(params, inds_query) # load representations to query for

        # Narrow search size by galaxy magnitude
        zmag_min = zmag[inds_query[0]] - params['delta_mag']
        zmag_max = zmag[inds_query[-1]] + params['delta_mag']

        ind_min, zmag_min = find_nearest(zmag, zmag_min)
        ind_max, zmag_max = find_nearest(zmag, zmag_max)

        chunk_min = ind_min // params['chunksize']
        chunk_max = math.ceil(ind_max / params['chunksize'])
        print(f'Magnitude range runs from index {ind_min}-{ind_max} (chunks {chunk_min}-{chunk_max})')

        # Now perform similarity search chunk by chunk
        # will return distance and index arrays of size (nquery, k)
        similarity_indexes, distance = calculate_similarity_for_chunk(params, rep_query, chunk_min, chunk_max)

        np.save(inds_file_out, similarity_indexes)
        np.save(dist_file_out, distance)


if __name__ == '__main__':

    args = parse_arguments()

    main(args)
