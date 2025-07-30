# Define the project name (use this for both the project and folder)
export PROJECT_NAME="gemm"  # This will represent both project name and folder name
export IP=132.236.59.64

# Run the Python script
python $PROJECT_NAME/$PROJECT_NAME.py

# Create the deploy folder and copy necessary files into it
mkdir -p deploy/$PROJECT_NAME   # No .prj for the folder
cp $PROJECT_NAME.prj/build_vivado/project_1.runs/impl_1/project_1_bd_wrapper.bit deploy/$PROJECT_NAME/vvadd.bit
cp $PROJECT_NAME.prj/build_vivado/project_1.gen/sources_1/bd/project_1_bd/hw_handoff/project_1_bd.hwh deploy/$PROJECT_NAME/vvadd.hwh
cp $PROJECT_NAME/host.py deploy/$PROJECT_NAME/host.py

# SCP the deploy folder to the remote machine
scp -r deploy/$PROJECT_NAME/ xilinx@$IP:~

# SSH into the remote machine
ssh xilinx@$IP
