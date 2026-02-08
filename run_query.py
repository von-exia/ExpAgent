"""
Test script for single query
"""
from agent_interface import AgentInterface

def main(query, agent_type="reactree"):
    agent_interface = AgentInterface(agent_type)
    print(f"Testing with query: {query}")
    print("="*50)

    # Execute the query
    response = agent_interface.response(query)
    print("Execution Result:")
    print(response)
    print("="*50)

    # Visualize the ReAcTree (Only support ReAcTree Now)
    if agent_type == "reactree":
        print("Generating visualization for ReAcTree...")
        agent_interface.visualize(title=f"ReAcTree Visualization", save_path="./reactree.png")

if __name__ == "__main__":
    # Test query 1 from GAIA
    query = """What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?"""
    
    # Test query 2 from GAIA
    # query = "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."
    
    # Test query for calculator
    # query = "Compute (3.013 + 1.43 * 5.6346) * 212, rounding to neareast thousand"
    
    # Test query for file read/write and shell tools
    # import os 
    # cur_root = os.getcwd()
    # file_path = os.path.join(cur_root, "py_test.py") # Get absolute path of file
    # query = f"Write a new function that implements multiplication to {file_path}, also create test sample, then run this python file"
    
    agent_type = "reactree"
    # agent_type = "react"
    main(query, agent_type)
    