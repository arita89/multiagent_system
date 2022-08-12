from _00_configuration._00_settings import *
#--------------------------------
# CUDA check
#--------------------------------
def cudaOverview():
    """
    just and overview of current status of cpu/gpu
    """
    cuda = torch.cuda.is_available()
    if not cuda:
        print ("CUDA not available, running on cpu")
        return "cpu"
    else:
        print ("CUDA available")
        numDevice = torch.cuda.device_count()
        idDevice =torch.cuda.current_device()
        deviceName = torch.cuda.get_device_name(0)
        print (f"Number of Devices: {numDevice}")
        print (f"ID current Device {deviceName}: {idDevice}")
        print (f"\tcurrent GPU memory usage by tensors in bytes:{torch.cuda.memory_allocated(idDevice)}")
        print (f"\tcurrent GPU memory managed by caching allocator in bytes:{ torch.cuda.memory_reserved(idDevice)}")

        #Assign cuda GPU located at location '0' to a variable
        cuda0 = torch.device('cuda:0')
        return cuda0

#--------------------------------
# VARIOUS PRINTING 
#--------------------------------

def printif(content, printstat= True, n= 10):
    """
    prints the content if the printstat is true
    by default prints at most 10 lines (eg if a list is very long)
    the printstat can be given by a condition
    """
    if printstat:
        if isinstance(content, list):
            l = len(content)
            if not l:
                print ("empty")
            elif l< n:
                print (*content, sep = "\n")
            else:
                print (*[f"{i}) {c}" for i,c in enumerate(content[:int(n/2)])], sep = "\n")
                print ("...")
                print (*[f"{l-int(n/2)+i}) {c}" for i,c in enumerate(content[-int(n/2):])], sep = "\n")
        else:
            print (content)

def list_all_in(directory, n = 5, printstat = True):
    """
    list all subdirectories and files in given directory
    """
    printif(f"dir:{directory}",printstat)
    dirpath, dirnames, filenames = next(os.walk(directory))
    if len(dirnames)>0 & printstat:
        print (f'DIRECTORIES\n{"-"*15}')
        print (*[">"+d for d in dirnames], sep= "\n")
    else:
        printif("no directories",printstat)
    print ()
    l = len(filenames)
    if printstat:
        if (l>0 and l <20) or n>=l:
            print (f'FILES: {l}\n{"-"*15}')
            print (*[f" {pad(i,minl = len(str(l)))}) "+f for i,f in enumerate(filenames)], sep= "\n")
        elif l == 0: 
            print ("no files")
        else:
            print (f'FILES: {l}\n{"-"*15}')
            print (*[f" {pad(i,minl = len(str(l)))}) "+f for i,f in enumerate(filenames[:n])], sep= "\n")
            print ("...")
            print (*[f" {pad(l-n+i,minl = len(str(l)))}) "+f for i,f in enumerate(filenames[-n:])], sep= "\n")
    return dirpath, dirnames, filenames

#--------------------------------
# TIME AND DATE
#--------------------------------

def get_timestamp(sep= "-"):
    """
    returns current time as a string in format %hours%minutes%seconds
    used for saving files and directories 
    """
    now = datetime.datetime.now()
    time_obj = now.strftime("%Hh%Mm%Ss")
    return time_obj

def get_date(sep= "-"):
    """
    returns current date as a string in format %year%month%day
    used for saving files and directories 
    """
    date_obj= datetime.date.today()
    date_obj = date_obj.strftime("%Y%m%d")+sep
    return date_obj

#--------------------------------
# HANDLE DIRECTORIES
#--------------------------------


def create_numbered_subdirectory(directory,datestamp):
    """
    counts the existing subdirs of given directory
    creates a new numbered one
    """
    numsubdir = len(os.listdir(directory))
    new_sub_label = "%03d"%(numsubdir+1)
    NEW_DIR= directory+f"{new_sub_label}-{datestamp[:-1]}/"
    if not os.path.exists(NEW_DIR):
        Path(NEW_DIR).mkdir(parents=True, exist_ok=True)
    print (f"new directory created at {NEW_DIR}")
    return NEW_DIR

def delete_empty(directory, printstat = True):
    """
    delete empty subdirectories in a given directory 
    not recursive
    returns a set with the deleted paths
    prints eventually while deleting
    """
    deleted = set()
    walk = list(os.walk(directory))
    for current_dir, _, _ in walk[::-1]:
        if len(os.listdir(current_dir)) == 0:
            printif(f"deleted path: {current_dir}", printstat)
            os.rmdir(current_dir)
            deleted.add(current_dir)
    return deleted

def delete_empty_r(directory):
    """
    delete empty subdirectories in a given directory 
    recursive
    returns a set with the deleted paths
    """
    deleted = set()
    for current_dir, subdirs, files in os.walk(directory, topdown=False):
        still_has_subdirs = any(
            _ for subdir in subdirs
            if os.path.join(current_dir, subdir) not in deleted
        )
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)

    return deleted

#--------------------------------
# HANDLE TEXTS
#--------------------------------

def remove_punctuation(s):
    """
    evokes unicorns
    """
    return s.translate(str.maketrans('', '', string.punctuation))

def pad(string,p = "0", minl= 2):
    """
    to pad the strings as wanted, eg
    input: "7",p= "i",minl  = 4
    ouptut: iii7
    """
    l = len(str(string))
    if l < minl: 
        return p*(minl-l)+str(string)
    else:
        return str(string)

def pad(string,p = "0", minl= 2):
    l = len(str(string))
    if l < minl: 
        return p*(minl-l)+str(string)
    else:
        return str(string)

#--------------------------------
# BUILD GIF
#--------------------------------

import imageio
def build_gif(folder,title,search = "", fps=55,recursive = True,delete_tempFiles = True):
    """
    titleGif = build_gif (folder,title,filenames,fps=55)
    folder = folder where are the current images to put togheter 
    title = name of gif
    filenames = list of names of images
    """
    
    SAVE_GIFS = "../data/gifs/"
    Path(SAVE_GIFS).mkdir(parents=True, exist_ok=True)

    filenames = sorted(glob.glob(folder + "/**/*" +f"*{search}*", recursive=recursive))
    print (f"found {len(filenames)} images in folder : {folder}")
    titleGif = os.path.join(SAVE_GIFS, f'{title}.gif')
    with imageio.get_writer(titleGif, mode='I',fps = fps) as writer:
        for filename in tqdm(filenames):
            #print (filename)
            image = imageio.imread(filename)
            writer.append_data(image)

            if delete_tempFiles: 
                try: os.remove(filename)
                except Exception as e:
                    print (e)
    
    #return titleGif

print (f"Helper Functions import successful")
