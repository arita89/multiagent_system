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
# DECORATOR TO RUN FUN ONLY ONCE
#--------------------------------
# https://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper   
    
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

@run_once
def printonceif(content, printstat= True, n= 10):
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
def create_subfolder_with_timestamp(folder):
    date = get_date()
    ts = get_timestamp()
    SAVE_TEMP = os.path.join(folder, f"{date}{ts}")
    Path(SAVE_TEMP).mkdir(parents=True, exist_ok=True)
    print (SAVE_TEMP)
    return SAVE_TEMP

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


def delete_empty_r(directory, printstat = True):
    """
    delete empty subdirectories in a given directory 
    recursive
    returns a set with the deleted paths
    """
    deleted = set()
    for current_dir, subdirs, files in os.walk(directory, topdown=False):
        still_has_subdirs = any(
            subdir for subdir in subdirs
            if os.path.join(current_dir, subdir) not in deleted
        )
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)
            printif(f"deleted path: {current_dir}", printstat)

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
## FORMATTING NAMES SAVINGS
#--------------------------------
def clean_string(mystring, lowercase = True):
    """
    modify as needed...
    replaces " " with "_" and " - " with "-"
    makes strings lowercase
    """
    mystring = mystring.replace("'", "")
    mystring = mystring.replace(", ", "_")
    mystring = mystring.replace("(", "-")
    mystring = mystring.replace(")", "-")
    mystring = mystring.replace("[", "_")
    mystring = mystring.replace("]", "_") 
    mystring.translate(str.maketrans('', '', string.punctuation))
    if lowercase == True:
        mystring = mystring.lower()
    return mystring 

def list2string(input_columns):
    return clean_string(str(input_columns))

#--------------------------------
## CHECK IF YOU ARE RUNNING COMBINATIONS AND IN THAT CASE JUST RUN WITHOUT ASKING INPUTS
#--------------------------------
def set_automatic_run(run_unattended,l):
    if run_unattended == False and l> 1:
        print ("Given that multiple combinations are going to be computed")
        print ("temporaly files will be automatically deleted at the end of each training without need of any intervention")
        run_unattended = True
    return run_unattended

#--------------------------------
# FLATTEN LIST of LISTS
#--------------------------------
# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
#def flatten(t):
    #return [item for sublist in t for item in sublist]
def list_flatten(t):
    return [item for sublist in t for item in sublist]

#--------------------------------------------------------------
# CHECK IF A KEY EXISTS IN A DICTIONARY, IF SO INCREASE THE NUMBERING
#-------------------------------------------------------------- 

def update_dict_with_key(this_key,this_value, this_dict):
    all_keys = list(this_dict.keys())
    count_occurrencies = sum(f'{this_key}' in s for s in all_keys)
    #print (count_occurrencies)
    if count_occurrencies == 0:
        index_label = ""
    else:
        index_label = pad(f"{count_occurrencies+1}",minl = 3)
        index_label = f"_{index_label}"
    this_dict[f"{this_key}{index_label}"] = this_value
    return this_dict

def update_dict_many_keys(new_dict, this_dict):
    for k,v in new_dict.items():
        this_dict = update_dict_with_key(k,v,this_dict)   
    return this_dict


def get_combo_dict(combo,combo_labels = None):
    if combo_labels is None:    
        combo_labels = [    
                        "reduction",# = combo[0]
                        "batch_size",# = combo[1]
                        "select_optimizer",# = combo[2]
                        "select_criterion",# = combo[3]
                        "hidden_layers_sizes",# = combo[4]
                        "lr",# = combo[5]
                        "momentum",# = combo[6]
                        "weight_decay",# = combo[7]
                        "select_scheduler",# = combo [8]
                        "use_edges_attr",# = combo [9]
                        "activation_fun",# = combo [10]
                    ]
    assert len(combo) == len (combo_labels)
    return dict(zip(combo_labels,combo))

#--------------------------------
# BUILD GIF
#--------------------------------

import imageio
def build_gif(folder,
              title,
              search = "", 
              fps=55,
              recursive = True,
              delete_tempFiles = True,
              max_n_images = 200
             ):
    """
    titleGif = build_gif (folder,title,filenames,fps=55)
    folder = folder where are the current images to put togheter 
    title = name of gif
    filenames = list of names of images
    """
    
    SAVE_GIFS = "../004_data/gifs/"
    Path(SAVE_GIFS).mkdir(parents=True, exist_ok=True)

    filenames = sorted(glob.glob(folder + "/**/*" +f"*{search}*", recursive=recursive))
    max_limit = min(max_n_images,len(filenames))
    print (f"found {len(filenames)} images in folder : {folder}")
    print (f"the gif will be create using the first {max_limit} images")
    titleGif = os.path.join(SAVE_GIFS, f'{title}.gif')
    with imageio.get_writer(titleGif, mode='I',fps = fps) as writer:
        for i,filename in tqdm(enumerate(filenames[:max_limit])):
            try: 
                image = imageio.imread(filename)
                writer.append_data(image)
            except Exception as e:
                print (e)

            if delete_tempFiles: 
                try: os.remove(filename)
                except Exception as e:
                    print (e)
                    
        if delete_tempFiles:            
            deleted_folders = delete_empty_r(directory= "../004_data/figures",
                                                 printstat = True)
    return titleGif

print (f"Helper Functions import successful")
