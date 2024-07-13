# HELPERS


def save_file(dir_name: str, file_name: str, args: str):
    '''
    Saves the output to given directory
    
    dir_name: assign directory name
    
    file_name: define the file name before passing to save_file
    
    args: assign city name: str
    
    '''
    import os
    from datetime import datetime
    import matplotlib.pyplot as plt
        
    try:
        timestamp = datetime.now().strftime("%d%m%y_%H%M")
        base_filename = f"{args}_{file_name}_{timestamp}.png"
        print(base_filename)
        
        # Check if the file exists and increment the counter if needed
        counter = 1
        final_filename = base_filename
        while os.path.exists(f"{dir_name}/{final_filename}"):
            final_filename = f"{args}_{file_name}_{timestamp}_{counter}.png"
            print(final_filename, "with counter")
            counter += 1
        
        # Save the plot to the data directory
        plt.savefig(f'{dir_name}/{final_filename}')
        plt.close()
        
    except Exception as e:
        print(f"save_file ERROR: {e}")