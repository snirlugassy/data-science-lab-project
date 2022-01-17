import sys

if __name__ == '__main__':
    print('Merging CSV files')
    input_files = sys.argv[1:-1]
    output_file = sys.argv[-1]

    with open(input_files[0], 'r') as f:
        headers = f.readline()

    with open(output_file, 'w') as output:
        output.write(headers)
        for file in input_files:
            print(f'Starting to append {file}')
            with open(file, 'r') as infile:
                # skip header
                infile.readline()
                output.writelines(infile.readlines())
            print(f'Finished appending {file}')
