import pycurl
import os
from multiprocessing import Process
def download_url(url, filename):
    fp = open(filename, "wb")
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, fp)
    c.perform()
    length = c.getinfo(c.CONTENT_LENGTH_DOWNLOAD)
    c.close()
    return length

def download_chunks(file_path, frame_count, download_path):
    length = 0
    maskfile=os.path.join(file_path, f'mask_{frame_count}.ncrf')
    maskfile_dl=os.path.join(download_path, f'mask_{frame_count}.ncrf')
    jsonfile=os.path.join(file_path, f'newheader_{frame_count}.json')
    jsonfile_dl=os.path.join(download_path, f'newheader_{frame_count}.json')
    deform=os.path.join(file_path, f'deform_{frame_count}.npy')
    deform_dl=os.path.join(download_path, f'deform_{frame_count}.npy')
    length = length + download_url(maskfile, maskfile_dl)
    length = length + download_url(jsonfile, jsonfile_dl)
    length = length + download_url(deform, deform_dl)
    # process_list = []
    # for i in range(13):
    #     ncrffile=os.path.join(file_path, f'2new_{frame_count}.ncrf{i}')
    #     ncrffile_dl=os.path.join(download_path, f'2new_{frame_count}.ncrf{i}')
    #     p = Process(target=download_url,args=(ncrffile, ncrffile_dl))
    #     p.start()
    #     process_list.append(p)

    # for i in process_list:
    #     p.join()
    for i in range(0, 13):
        ncrffile=os.path.join(file_path, f'2new_{frame_count}.ncrf{i}')
        ncrffile_dl=os.path.join(download_path, f'2new_{frame_count}.ncrf{i}')
        tmp = download_url(ncrffile, ncrffile_dl)
        length = tmp + length
    return length