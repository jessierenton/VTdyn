#Code for Voronoi Tessellation model

code for running simulating invasion processes in the Voronoi tessellation model. Main code is divided into three subdirectories: structure, libs and run_files.
- structure contains several modules needed to create and update a Tissue object, which represents the VT model at a single timestep
	- cell.py: defines Tissue (has a Mesh and a Force as attributes, as well as individual cell information, such as ancestory, age, type etc. Also methods for updating the tissue, e.g. cells moving according to the force law, cell division and death.) and Force objects. 
	- mesh.py: defines Mesh object(spatial information for the tissue including positions of cells, neighbour information, and methods for updating)
	- global_constants.py: defines VT model parameters
	- initialisation.py: functions for creating an initial Tissue

- libs contains modules with functions for running simulations
	- pd_lib.py is for the additive prisoner's dilemma with decoupled or death-birth update rules
	- public_goods_lib.py is for arbitrary multiplayer games with decoupled or death-birth update rules
	- contact_inhibition_lib.py is for the additive prisoner's dilemma with seperate birth and death processes, where birth is only allowed above an area threshold
	- plot.py contatins plotting routines (torus_plot and animate torus are most useful)
	- data.py contains useful data manipulation routines
- run_files contains various files for running simulations that import simulation routines from a given lib. These need to be moved into the main file to be run. Examples:
	- pd_original/multirun_constant_N.py and d_original/multirun_death_birth.py are used for running simulations of the additive prisoner's dilemma (cooperator invasion) with decoupled and death-birth update rule respectively.
	- cip_area_threshold/run_CIP_parallel_pd.py can be used for running simulations of the additive prisoner's dilemma with contact inhibition
	
More detailed instructions for running simulations of the additive prisoner's dilemma or public goods games with decoupled/death-birth update rules can be found in the repositories evo-epithelium and pgg-epithelium, respectively. These also contain data obtained from simulations. 
