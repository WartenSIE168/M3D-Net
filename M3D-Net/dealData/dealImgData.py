import os
import numpy as np
import pandas as pd

'''
Slice image text according to the time in text files.
The slicing range is from 6 to 95.
Get the number of labels.
Process multiple files.
Save the processed data in the processData/dealImgData directory.
After processing the image files, they are still in TXT format, but the second column is modified to the image paths.
'''
class DealImgDataDemo():
    def __init__(self):
        super(DealImgDataDemo, self).__init__()
        self.data_row = 100  # Minimum data length

    def slice_and_process_file(self, input_file, line_numbers):
        try:
            # Get file information
            # file_dir = os.path.dirname(os.path.abspath(input_file))
            file_name = os.path.basename(input_file)
            base_name, ext = os.path.splitext(file_name)

            # Generate output file name
            output_file = os.path.join('..\\processData\\dealImgData', f"{base_name}_processed{ext}")

            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(input_file))

            # Read all lines of the original file
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            processed_lines = []

            # Process 5 lines starting from each starting line number
            for start_line in line_numbers:
                start_line += 6
                end_line = start_line + self.data_row - 10
                # Ensure not to exceed the file range
                if start_line < len(lines):
                    selected_lines = lines[start_line:end_line]

                    for line in selected_lines:
                        # Split line data (using comma as separator)
                        columns = line.strip().split(',')
                        if len(columns) >= 2:  # Ensure there is a second column
                            # Modify the second column: current directory + original second column data
                            columns[1] = os.path.join(current_dir, columns[1])
                            # Recombine the line
                            processed_line = ','.join(columns) + '\n'
                            processed_lines.append(processed_line)
                        else:
                            # Keep lines without a second column unchanged
                            processed_lines.append(line)

            # Write to the new file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)

            print(f"File processing completed, a total of {len(processed_lines)} lines were processed.")
            print(f"Original file: {input_file}")
            print(f"Processed file: {output_file}")

            return output_file

        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    # Get all txt files in the current image directory
    def get_txt_files(self, directory=None):
        """
        Get all .txt files in the specified directory and its subdirectories.

        Parameters:
            directory (str): The directory path to search, default is the current directory.

        Returns:
            list: A list containing the full paths of all .txt files.
        """
        if directory is None:
            directory = os.getcwd()  # Get the current working directory

        txt_files = []
        files_name = []

        # Traverse the directory tree
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    # Get the full path of the file
                    full_path = os.path.join(root, file)
                    txt_files.append(full_path)
                    files_name.append(file)

        return txt_files, files_name

    # Load txt file and get data
    def load_eye_txt(self, files):
        """Read a comma-separated text file into a numpy array."""
        with open(files, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
        data = []
        for line in lines:
            # Filter empty lines and split data
            if line.strip():
                row = [int(x.strip()) for x in line.split(',')]
                data.append(row)
        return np.array(data)

    # Get the number of targets and labels in the eye movement file
    def get_exp_num(self, files):
        eye_ = self.load_eye_txt(files)
        current_key = None
        current_length = 0
        exp_num = []        # Number of targets
        diff_line = []      # Different positions
        labels = []         # Labels
        # diff_line.append(0)   # First line
        # exp_num.append(eye_[0, 0])
        # labels.append(eye_[0, -1])
        # Get the number of targets
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
            # if eye_[row - 1, 0] != eye_[row, 0]:
            #     exp_num.append(eye_[row, 0])
            #     labels.append(eye_[row, -1])
        return exp_num, diff_line, labels

    def find_matching_indices(self, list1, list2):
        """
        Find the indices of equal elements in two lists.

        Parameters:
            list1 (list): The first list (no duplicate elements).
            list2 (list): The second list.

        Returns:
            list: A list containing tuples, each tuple is (element, index in list1, index in list2).
        """
        # Create a dictionary to store elements and their corresponding indices in list1
        element_to_index = {element: index for index, element in enumerate(list1)}

        result = []

        for index2, element in enumerate(list2):
            if element in element_to_index:
                index1 = element_to_index[element]
                result.append((element, index1, index2))

        return result

    def count_2d_elements_pandas(self, matrix):
        """
        Count elements in a 2D list using pandas.
        """
        # Flatten the 2D list
        flattened = [item for sublist in matrix for item in sublist]
        # Convert to Series and count
        return pd.Series(flattened).value_counts().to_dict()

# Usage example
if __name__ == "__main__":
    labels = []
    dealImg = DealImgDataDemo()
    img_files, img_name = dealImg.get_txt_files('..\\rowData\\image')      # Get image paths
    txt_files, txt_name = dealImg.get_txt_files('..\\rowData\\eye_data')   # Get eye movement text data
    # print(img_files, txt_files)
    matches = dealImg.find_matching_indices(img_name, txt_name)
    print("Matching results (element, index in list1, index in list2):")
    print(matches)
    # _, diff_line, label = dealImg.get_exp_num(txt_files[matches[0][2]])
    # print(diff_line, label)
    for match in range(len(matches)):
        _, diff_line, label = dealImg.get_exp_num(txt_files[matches[match][2]])
        print(label, len(label))
        dealImg.slice_and_process_file(img_files[matches[match][1]], diff_line)     # Process image files
        labels.append(label)
    count = dealImg.count_2d_elements_pandas(labels)
    print(count)