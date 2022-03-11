# Author: Christoph Menz
import os, shutil
import subprocess
import numpy as np
import pandas as pd
import mpi4py.MPI
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xml.etree.ElementTree as ET

# ############################################################################## 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mkdir(dir = None):
  '''
  Creates the given directory.

  Parameters
  ----------
  dir : char
           Directory
  '''
  if not dir is None:
    if not os.path.exists(dir):
      os.makedirs(dir)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def clean_up(path):
    if os.path.exists(path):
        shutil.rmtree(path)      
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
def create_simulation_environment(
    STICS_path = None,
    STICS_bin = 'JavaSticsCmd.exe',
    STICS_local_path = None,
    link_items = None,
    copy_items = None):
    # default values
    if link_items is None:
        link_items = ['bin',STICS_bin]
    if copy_items is None:
        copy_items = [
                      'config','grape\\CLIMAISJ.2011','grape\\CLIMAISJ.2012','grape\\CLIMAISJ.2013','grape\\CLIMAISJ.2014','grape\\CLIMAISJ_sta.xml',
                      'grape\\mais_ini.xml','grape\\mais.lai','grape\\Mais_tec.xml','grape\\prof.mod','grape\\sols.xml','grape\\usms.xml', 'grape\\var.mod', 'grape\\rap.mod',
                     ]

    # clean up directory, just to be sure
    clean_up(STICS_local_path)

    # link and copy files
    for item in link_items + copy_items:
        # check if the parent directory exists
        dirname = os.path.dirname(os.path.join(STICS_local_path, item)) # obtain the corresponding local directory name where the item was located
        if not dirname == '' and not os.path.exists(dirname):
            print('  %Status: directory {0} does not exist and will be created.'.format(dirname))
            mkdir(dir = dirname) # create STICS_local_path directory and subdirectory as workspace 
        if item in link_items:
            os.symlink(os.path.join(STICS_path, item), os.path.join(STICS_local_path,item)) # create a symbolic link from source (1st arg) to destination (2nd arg)
        elif item in copy_items:
            if os.path.isfile(os.path.join(STICS_path, item)):
                shutil.copy(os.path.join(STICS_path, item), os.path.join(STICS_local_path, item)) # if the item is a file, copy it to destination folder
            else:
                shutil.copytree(os.path.join(STICS_path, item), os.path.join(STICS_local_path, item)) # if the item is a directory, copy the whole directory tree to the destination folder
    #shutil.copy(os.path.join(STICS_path, STICS_bin), os.path.join(STICS_local_path,STICS_bin)) # copy cmd binary file from source (1st arg) to destionation (2nd arg)
    # add plant dir to hold plat parameter files
    mkdir(os.path.join(STICS_local_path, 'plant'))
	
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ############################################################################## 
comm = mpi4py.MPI.COMM_WORLD

# 1. User-specific session 
# =============================================================================
Root_path    = "G:\SpatialModellingSRC\STICS_PIK_Cluster" # Define the root path 
CurrentDir   = os.getcwd() # Specify the current working directory where EF function has been stored
STICS_path   = os.path.join(Root_path, 'STICS_V91') # Specify the STICS main directory
STICS_bin    = 'JavaSticsCmd.exe' # Specify STICS running command directory 
STICS_workspace = 'grape' # Specify the STICS workspace folder name
start_year=2011
end_year=2014
Study_years=np.arange(start_year,end_year+1,1)
Climate_input=["CLIMAISJ."+str(int(year)) for year in Study_years]
STICS_workspace_files = Climate_input+['CLIMAISJ_sta.xml', # Concatenate climate input files with other essential input files
                         'mais_ini.xml','mais.lai','Mais_tec.xml','prof.mod','sols.xml','usms.xml', 'var.mod', 'rap.mod'] # 'mais.lai',
STICS_workdir = os.path.join(STICS_path, 'multiprocess') # Create a folder that enable running the MPI multiprocess
variety = 'Touriga_Nacional' # Specify the name of variety node that is appearing in the plant .xml file
STICS_plant_default=os.path.join(STICS_path, 'plant', 'vine_TN_plt_default.xml') # Default plant file for variety TN used to read default configuration every time
STICS_plant_file='vine_TN_plt.xml' # Given a plant name
fname_out = os.path.join(Root_path,'cali_{0}.dat'.format(variety.lower())) # Define the output file name

# 2 Supply observational data
Observation_file = os.path.join(STICS_path, 'Observation.xlsx') # Specify the observational file that contain the measurements of variables
LoadOB=pd.ExcelFile(Observation_file)
ListofSheets=LoadOB.sheet_names # Obtain the information on excel sheets 
Ob_dataframe=pd.read_excel(LoadOB,sheet_name=ListofSheets[0]) # read observational data into a pd.df
Ob_dataframe.columns=['Year','DOYFlower','DOYHarvest','Yield'] # Rename the column name of df
OB_DOYFlower=Ob_dataframe['DOYFlower'].astype(float) # Change datatype of data column to float
OB_DOYHarvest=Ob_dataframe['DOYHarvest'].astype(float)
OB_Yield=Ob_dataframe['Yield'].astype(float)
# Create the observational dictionary
OB = {'Flower':Ob_dataframe['DOYFlower'].astype(float),
      'Harvest':Ob_dataframe['DOYHarvest'].astype(float),
      'Yield':Ob_dataframe['Yield'].astype(float),
     }
     
# 3. Define the STICS model parameters to be calibrated (mostly plant file) along with their specified ranges
# =============================================================================
# 3.1 Create parameter lists with specified range of values to be tested
param_lib = {
             'FruitsettingGDD':np.arange(50,425,75).tolist(), 
             'FruitfillingGDD':np.arange(700,1700,200).tolist(), 
             'ReproGDD':np.arange(250,500,50).tolist(),
             'GrainNumberPerGDD':np.arange(0.5,3,0.5).tolist(), 
             'VeraisonGDD':np.arange(600,1600,200).tolist(), 
             'Boxnumber':np.arange(5,20,5).tolist(), 
             'WaterDynamicGDD':np.arange(100,350,50).tolist(), 
			 'GrainWeight':np.arange(0.5,2.0,0.3).tolist(),
	         'SourceSinkRatio':np.arange(0.25,1.0,0.25).tolist()
            }

param_prefix = {
             'Boxnumber':'BN',
             'FruitsettingGDD':'FS',
             'FruitfillingGDD':'FF',
             'ReproGDD':'RG',
             'GrainNumberPerGDD':'GN',
             'VeraisonGDD':'VG',
             'WaterDynamicGDD':'WD',
			 'GrainWeight':'GW',
			 'SourceSinkRatio':'SS'
            }

# =============================================================================

# 3.2  Parallel running all possible parameter combinations and compute several statistics per combination

MSE_dict={} # Create an empty dictionary to store computed statistical results for MSE
SSE_dict={} # Create an empty dictionary to store computed statistical results for SSE
nMAE_dict={} # Create an empty dictionary to store computed statistical results for nMAE
MAPE_dict={} # Create an empty dictionary to store computed statistical results for MAPE
nRMSE_dict={} # Create an empty dictionary to store computed statistical results for nRMSE
R2_dict={} # Create an empty dictionary to store computed statistical results for R2
Bug_dict={} # Create an empty dictionary to catch results of parameter vector resulting in errors or bugs during the simulations 
# Initiate mpi instance
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
# disentangle param_lib into keys and values
param_keys, param_values = zip(*param_lib.items())
# Start the paralllel process where each cpu will handle one parameter combination per time
for index, param in enumerate(itertools.product(*param_values)):

    if index%mpi_size == mpi_rank:
        print('  %Status: This is the iteration loop: ' + str(index+1) + ' run by rank ' + str(mpi_rank) + '#############################################################################')
        # build param dict
        param = dict(zip(param_keys, param))

        # 3.2.1 Obtain the parameter vector label in the curret iteration loop
        #ParaVector = ':'.join([param_prefix[key] + str(param_lib[key].index(param[key])) for key in param_lib.keys()])
        ParaVector = ':'.join([param_prefix[key] + str(param[key]) for key in param_lib.keys()])

        # build local simulation environment (according to cpu rank)
        # mpi_rank == 0 : grape_0000
        # mpi_rank == 1 : grape_0001
        # mpi_rank == 2 : grape_0002
        # ...
        STICS_local_path  = os.path.join(STICS_workdir, STICS_workspace + '_{0:04d}'.format(mpi_rank))

        STICS_outputfile  = os.path.join(STICS_local_path,STICS_workspace,'mod_rapport.sti') # Specify the path to STICS output report file
        STICS_local_plant = os.path.join(STICS_local_path,'plant',STICS_plant_file) #Varierty corresponds to TN

        create_simulation_environment(STICS_path = STICS_path, STICS_bin = STICS_bin, STICS_local_path = STICS_local_path,
                                      copy_items = ['config'] +[STICS_workspace + '\\' + fname for fname in STICS_workspace_files])

        # 3.2.2 Set plant parameters with testing values
        variety_param_node = "formalisme[@nom='cultivar parameters']/tv[@nom='genotypes']/variete[@nom='{0}']".format(variety)
        general_param_node = "formalisme[@nom='yield formation']/option[@nom='growing dynamics']/choix[@nom='indeterminate growing plant']" 

        xml_tree = ET.parse(STICS_plant_default)
        xml_root = xml_tree.getroot()
        xml_root.findall(variety_param_node + "/optionv[@nom='codeindetermin']/param[@nom='dureefruit']")[0].text = str(param['FruitfillingGDD'])
        xml_root.findall(variety_param_node + "/optionv[@nom='codeindetermin']/param[@nom='afruitpot']")[0].text  = str(param['GrainNumberPerGDD'])
        xml_root.findall(variety_param_node + "/param[@nom='stlevdrp']")[0].text                                  = str(param['ReproGDD'])
        xml_root.findall(variety_param_node + "/param[@nom='stamflax']")[0].text                                  = str(param['VeraisonGDD'])
        xml_root.findall(variety_param_node + "/param[@nom='stdrpdes']")[0].text                                  = str(param['WaterDynamicGDD'])
        xml_root.findall(variety_param_node + "/param[@nom='pgrainmaxi']")[0].text                                = str(param['GrainWeight'])
        xml_root.findall(general_param_node + "/param[@nom='stdrpnou']")[0].text                                  = str(param['FruitsettingGDD'])
        xml_root.findall(general_param_node + "/param[@nom='nboite']")[0].text                                    = str(param['Boxnumber'])
        xml_root.findall(general_param_node + "/param[@nom='spfrmin']")[0].text                                   = str(param['SourceSinkRatio'])		
        xml_tree.write(STICS_local_plant)
        # ================================================================= ============
        # 3.2.3 Run STICS model with each combined parameter vector
        os.chdir(STICS_local_path) # It is necessary to change the current python console working directory to STICS model main directory 
        # The execution command is for the whole STICS directory USMs
        # In this case, USM xml file should be properly adjusted and aligned to number of observational years
        STICS_runCMD=os.path.join(STICS_local_path,STICS_bin)+ " " + "--run" + " " + STICS_workspace
        try:
            stream = subprocess.Popen(STICS_runCMD)
            stream.wait()
        except:
            Bug_dict.update({ParaVector:'NaN'}) # catch the erroneous parameter vector
            clean_up(STICS_local_path)
            continue
        os.chdir(CurrentDir) # Change directory back to the current python console directory
        #output
        # =============================================================================
        # 3.2.4 Read STICS output file and process the output data 
        try:
            Outputdata = pd.read_csv(STICS_outputfile, header = 0, delimiter = ';', index_col = False, skiprows = lambda i: i>0 and i%2==0)
        except:
            Bug_dict.update({ParaVector:'NaN'}) # catch the erroneous parameter vector
            clean_up(STICS_local_path)
            continue
        # Create the simulation dict
        SM = {'Flower':Outputdata['iflos'].astype(float)-365, # The output column is indexed by the header name
              'Harvest':Outputdata['irecs'].astype(float)-365, # The output column is indexed by the header name
              'Yield':Outputdata['mafruit_kg_ha'].astype(float)/0.23 # The output column is indexed by the header name, convert dry yield to fresh productivity at 77% water content (maturity criteria)
             }

        os.remove(STICS_outputfile) # Delete the output report file so that the next iteration can generate new data
        # =============================================================================
        # 3.2.5 Compare STICS simulations with observations and derive statistics of prediction performance
        # The calibration approach is adopted using one of frequentist model: not independent errors, several response variables and non-zero expectation
        # The assigned weight for the computed statistics reflected one criteria: more weight imposed on the response variable, higher priority to improve prediction accuracy on that variable                
        try:
            MSE   = {key:mean_squared_error(OB[key],SM[key]) for key in OB.keys()}
            SSE   = {key:MSE[key]*len(OB[key]) for key in OB.keys()}
            nMAE  = {key:mean_absolute_error(OB[key],SM[key])/np.mean(OB[key]) for key in OB.keys()}
            MAPE  = {key:mean_absolute_percentage_error(OB[key],SM[key]) for key in OB.keys()}
            nRMSE = {key:np.sqrt(MSE[key])/np.mean(OB[key]) for key in OB.keys()}
            R2    = {key:r2_score(OB[key],SM[key]) for key in OB.keys()}
        except:
            Bug_dict.update({ParaVector:'NaN'}) # catch the erroneous parameter vector
            clean_up(STICS_local_path)
            continue

        MSE_dict[ParaVector] = MSE
        SSE_dict[ParaVector] = SSE
        nMAE_dict[ParaVector] = nMAE
        MAPE_dict[ParaVector] = MAPE
        nRMSE_dict[ParaVector] = nRMSE
        R2_dict[ParaVector] = R2
        # =============================================================================
        clean_up(STICS_local_path) # Clean up the path after each cpu computes the required statistics


# 4. Gather all data at mpi_rank == 0 and write the results to csv file
mpi_package = {'SSE':SSE_dict, 'nMAE':nMAE_dict, 'MAPE':MAPE_dict, 'nRMSE':nRMSE_dict, 'MSE':MSE_dict, 'R2':R2_dict}
if mpi_rank != 0:
    comm.send(mpi_package, dest = 0, tag = mpi_rank)
else: # performed by CPU=0
    full_package = [mpi_package]
    for isender in range(1,mpi_size):
        pack = comm.recv(source = isender, tag = isender)
        full_package.append(pack)
    # 4.1 Gather all statistical results into a score lib
    # final evaluation is done by mpi_rank == 0
    # build score_lib with all scores
    score_lib = {}
    for pack in full_package:
        for key in pack.keys():
            for item in pack[key].keys():
                for variable in pack[key][item].keys():
                    if not key+'_'+variable in score_lib.keys():
                        score_lib[key+'_'+variable] = {}
                    score_lib[key+'_'+variable][item] = pack[key][item][variable]
    # 4.2 Save results
    # =========================================================================================================
    pd.DataFrame(score_lib).to_csv(fname_out, sep = ' ', index_label = 'parameter')
