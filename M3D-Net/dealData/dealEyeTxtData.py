import numpy as np
import warnings
import os

'''
Read the visual information text data and perform slicing processing.
The slicing range is from 6 to 95.
Extract data from different files to build a dataset and labels.
Save all the extracted data to the corresponding files in the processData\\dealEyeData directory.
'''

warnings.filterwarnings("ignore")


class ReadDataDemo():
    def __init__(self):
        super(ReadDataDemo, self).__init__()

        self.data_row = 100

    # Load the txt file and get the data
    def load_eye_txt(self, file):
        """Read a comma-separated text file into a numpy array"""
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
        data = []
        for line in lines:
            # Filter empty lines and split the data
            if line.strip():
                row = [int(x.strip()) for x in line.split(',')]
                data.append(row)
        return np.array(data)

    # Get all txt files in the current image directory
    def get_txt_files(self, directory=None):
        """
        Get all .txt files in the specified directory and its subdirectories.

        Parameters:
            directory (str): The directory path to search. Defaults to the current directory.

        Returns:
            list: A list containing the full paths of all .txt files.
        """
        if directory is None:
            directory = os.getcwd()  # Get the current working directory

        txt_files = []
        txt_name = []

        # Traverse the directory tree
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    # Get the full path of the file
                    full_path = os.path.join(root, file)
                    txt_files.append(full_path)
                    txt_name.append(file)

        return txt_files, txt_name

    # Get the number of targets and labels in the eye movement file
    def get_exp_num(self, eye_):
        current_key = None
        current_length = 0
        exp_num = []  # Number of targets
        diff_line = []  # Different positions
        labels = []  # Labels

        for row in range(1, len(eye_)):
            if current_key is None:
                current_key = eye_[row - 1, 0]
                current_length = 1
            elif eye_[row, 0] == current_key:
                current_length += 1
            else:
                if current_length > self.data_row:
                    exp_num.append(eye_[row - 1, 0])
                    labels.append(eye_[row - 1, -1])
                    diff_line.append(row - current_length)
                current_key = eye_[row, 0]
                current_length = 1

        return exp_num, diff_line, labels

    def slice_and_process_file(self, input_file, file_name):
        try:
            # Generate the output file name
            base_name, ext = os.path.splitext(file_name)
            output_file = os.path.join('..\\processData\\dealEyeData', f"{base_name}_processed{ext}")

            eye_ = self.load_eye_txt(input_file)

            # Get the number of targets in the eye movement file
            _, exp_diffline, _ = self.get_exp_num(eye_)

            # Read all lines of the original file
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            processed_lines = []

            # Process 5 lines starting from each starting line number
            for start_line in exp_diffline:
                start_line += 6
                end_line = start_line + self.data_row - 10
                # Ensure not to exceed the file range
                if start_line < len(lines):
                    selected_lines = lines[start_line:end_line]

                    for line in selected_lines:
                        # Split the line data (using comma as the separator)
                        columns = line.strip().split(',')
                        processed_line = ','.join(columns) + '\n'
                        processed_lines.append(processed_line)

            # Write to the new file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)

            print(f"File processing completed. A total of {len(processed_lines)} lines were processed.")
            print(f"Original file: {input_file}")
            print(f"Processed file: {output_file}")

            return output_file

        except FileNotFoundError:
            print(f"Error: The input file {input_file} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None


if __name__ == '__main__':
    readdata = ReadDataDemo()
    txt_files, txt_name = readdata.get_txt_files('..\\rowData\\eye_data')
    for i in range(len(txt_files)):
        print(txt_files[i], txt_name[i])
        readdata.slice_and_process_file(txt_files[i], txt_name[i])
    