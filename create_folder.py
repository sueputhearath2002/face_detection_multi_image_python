import os
import json

# Ask user for the folder name
folder_name = input("Enter the folder name: ")

# Initialize response dictionary
response = {
    "folder_name": folder_name,
    "status": "",
    "message": ""
}

# Attempt to create the folder
try:
    os.mkdir(folder_name)
    response["status"] = "success"
    response["message"] = f"Folder '{folder_name}' created successfully!"
except FileExistsError:
    response["status"] = "error"
    response["message"] = f"Folder '{folder_name}' already exists."
except Exception as e:
    response["status"] = "error"
    response["message"] = str(e)

# Return the response as a JSON string
response_json = json.dumps(response, indent=4)
print(response_json)
