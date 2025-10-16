import os
import math
import shutil
import requests

 

# Use http://localhost:7878 if you run a local instance.

ADDOK_URL = """CHANGE WITH YOUR OWN SERVER ADRESS"""

 
def geocode(filepath_in, requests_options, filepath_out='geocoded.csv'):
    with open(filepath_in, 'rb') as f:
        filename, response = post_to_addok(filepath_in, f.read(), requests_options)
        # print(response)
        write_response_to_disk(filepath_out, response)
        


def geocode_chunked(filepath_in, filename_pattern, chunk_by_approximate_lines, requests_options):
    b = os.path.getsize(filepath_in)
    output_files = []
    with open(filepath_in, 'r') as bigfile:
        row_count = sum(1 for row in bigfile)
    with open(filepath_in, 'r') as bigfile:
        headers = bigfile.readline()
        chunk_by = math.ceil(b / row_count * chunk_by_approximate_lines)
        current_lines = bigfile.readlines(chunk_by)
        i = 1
        # import ipdb;ipdb.set_trace()
        while current_lines:
            current_filename = filename_pattern.format(i)
            current_csv = ''.join([headers] + current_lines)
            # import ipdb;ipdb.set_trace()
            filename, response = post_to_addok(current_filename, current_csv, requests_options)
            write_response_to_disk(current_filename, response)
            current_lines = bigfile.readlines(chunk_by)
            i += 1
            output_files.append(current_filename)
    return output_files

 

def write_response_to_disk(filename, response, chunk_size=1024):
    with open(filename, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def post_to_addok(filename, filelike_object, requests_options):
    files = {'data': (filename, filelike_object)}
    response = requests.post(ADDOK_URL, files=files, data=requests_options)
   # You might want to use https://github.com/g2p/rfc6266
    content_disposition = response.headers['content-disposition']
    filename = content_disposition[len('attachment; filename="'):-len('"')]
    return filename, response


def process_dataframe_in_chunks2(df, label ,chunk_size, chunks_dir_path, chunks_geo_dir_path, columns): 
    
    for i, chunk in enumerate(range(0, len(df), chunk_size)): 
    
        df_chunk = df.iloc[chunk:chunk + chunk_size] 
        chunk_path_in = os.path.join(chunks_dir_path, f"{label}_chunk_{i}.csv") 
        chunk_path_out = os.path.join(chunks_geo_dir_path, f"{label}_chunk_{i}_geocoded.csv") 
        df_chunk.to_csv(chunk_path_in, index=False)
        
        geocode(chunk_path_in, columns, chunk_path_out)

 

 
def consolidate_multiple_csv(files, output_name):
    with open(output_name, 'wb') as outfile:
        for i, fname in enumerate(files):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)


